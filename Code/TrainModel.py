import numpy as np
import pandas as pd
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.data import Dataset
from tensorflow_addons.optimizers import AdamW
from transformers import BertTokenizer, TFBertForSequenceClassification
from time import strftime

from Helper.TensorflowPlus import SlantedTriangularLearningRate, ThreeClassAccuracy
from Helper.HyperparameterTuner import GridTuner

### Configuration ###

# File Configuration
train_file = r'../Data/Human Annotated Data/NLP.json'
dev_file = r'../Data/Human Annotated Data/ML.json'

# Hyperparameter Configuration
hyperparameters = {
	'batch_size': [16],
	'epochs': [3],
	'learning_rate': [5e-5],
	'warmup_ratio': [0.06],
}
executions_per_trial = 10


### Setup ###

# Load data and tokenizer
dataset = {
	'train': pd.read_json(train_file),
	'dev': pd.read_json(dev_file),
}
tokenizer = BertTokenizer.from_pretrained('allenai/scibert_scivocab_uncased', do_lower_case=True)

# Tokenize papers
encoded_dataset = dict()
for t in dataset:
	data = dataset[t].apply(lambda p: p['title'] + '[SEP]' + p['abstract'], axis=1).to_list()
	encodings = tokenizer(data, max_length=300, padding='max_length', truncation=True)
	encoded_dataset[t] = Dataset.from_tensor_slices((dict(encodings), dataset[t]['stance']))


### Model Builder ###

def build_model(hp):
	# load model
	model = TFBertForSequenceClassification.from_pretrained('allenai/scibert_scivocab_uncased', num_labels=1, from_pt=True)
	
	# get parameters
	batch_size = hp.get('batch_size', values=hyperparameters['batch_size'])
	epochs = hp.get('epochs', values=hyperparameters['epochs'])
	learning_rate = hp.get('learning_rate', values=hyperparameters['learning_rate'])
	warmup_ratio = hp.get('warmup_ratio', values=hyperparameters.get('warmup_ratio', [0.0]))
	weight_decay = hp.get('weight_decay', values=hyperparameters.get('weight_decay', [0.0]))
	adam_epsilon = hp.get('adam_epsilon', values=hyperparameters.get('adam_epsilon', [1e-6]))
	adam_beta_1 = hp.get('adam_beta_1', values=hyperparameters.get('adam_beta_1', [0.9]))
	adam_beta_2 = hp.get('adam_beta_2', values=hyperparameters.get('adam_beta_2', [0.999]))
	
	# build model
	lr_schedule = SlantedTriangularLearningRate(
		maximum_learning_rate=learning_rate,
		number_of_iterations=int(epochs*len(encoded_dataset['train'])/batch_size),
		cut_frac=warmup_ratio,
	)
	optimizer = AdamW(
		learning_rate=lr_schedule,
		weight_decay=weight_decay,
		epsilon=adam_epsilon,
		beta_1=adam_beta_1,
		beta_2=adam_beta_2,
	)
	model.compile(
		optimizer=optimizer,
		loss=MeanSquaredError(),
		metrics=[ThreeClassAccuracy()],
	)
	return model


### Train Model ###

# Create grid tuner
tuner = GridTuner(
	build_model,
	objective='val_loss',
	direction='minimize',
	executions_per_trial=executions_per_trial,
	save_best_model_weights=True,
	output_dir='TrainModel/%s' % strftime('%Y-%m-%d_%H-%M-%S'),
)

# Start grid search
tuner.search(
	data=encoded_dataset['train'].shuffle(100000, reshuffle_each_iteration=True).batch(16),
	validation_data=encoded_dataset['dev'].batch(16)
)

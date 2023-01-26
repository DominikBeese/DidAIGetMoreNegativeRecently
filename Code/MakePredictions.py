import numpy as np
import pandas as pd
from tensorflow.data import Dataset
from transformers import BertTokenizer, TFBertForSequenceClassification

# Configuration
model_file = r'../Model/analysis-model.h5'
data_file = r'../Data/Datasets/NLP.json'
output_file = r'NLP-Predictions.json'

# Load data, tokenizer, model, and weights
dataset = pd.read_json(data_file)
tokenizer = BertTokenizer.from_pretrained('allenai/scibert_scivocab_uncased', do_lower_case=True)
model = TFBertForSequenceClassification.from_pretrained('allenai/scibert_scivocab_uncased', num_labels=1, from_pt=True)
model.load_weights(model_file)

# Tokenize papers
data = dataset.apply(lambda p: p['title'] + '[SEP]' + p['abstract'], axis=1).to_list()
encodings = tokenizer(data, max_length=300, padding='max_length', truncation=True)
encoded_dataset = Dataset.from_tensor_slices(dict(encodings))

# Make predictions
predictions = model.predict(encoded_dataset.batch(16), verbose=1).logits.tolist()

# Save predictions
dataset['stance'] = np.clip(np.array(predictions)[:,0], -1, 1)
dataset.to_json(output_file, orient='records', indent=2, force_ascii=False)

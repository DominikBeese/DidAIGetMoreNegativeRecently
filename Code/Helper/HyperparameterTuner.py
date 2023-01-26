import tensorflow as tf
import numpy as np
from os import makedirs, urandom
from os.path import join
import json

class Hyperparameters():
	def __init__(self, start_with_trial=0):
		self.names = list()
		self.values = list()
		self.trial = start_with_trial
	
	def __contains__(self, item):
		return item in self.names
	
	def get(self, name, values=None):
		if name not in self:
			self.names.append(name)
			self.values.append(values)
			return values[0]
		else:
			i = self.names.index(name)
			values = self.values[i]
			length = len(values)
			offset = np.prod([len(v) for v in self.values[:i]]) or 1
			return values[int(self.trial/offset)%length]
	
	def getAll(self):
		return {name: self.get(name) for name in self.names}
	
	def trials(self):
		return np.prod([len(v) for v in self.values])
	
	def nextTrial(self):
		self.trial += 1

class GridTuner():
	def __init__(self, build_model, objective, direction='minimize', executions_per_trial=1, output_dir=None, output_predictions=True, save_best_model_weights=False):
		self.build_model = build_model
		self.objective = objective
		self.extremum = min if direction == 'minimize' else max
		self.executions_per_trial = executions_per_trial
		self.output_dir = output_dir
		self.output_predictions = output_predictions
		self.save_best_model_weights = save_best_model_weights
		self.log = list()
		self.predictions = list()
	
	def search(self, data, start_with_trial=1, run_after_each_execution=None, **kwargs):
		""" Starts the full grid search.
			data - training data
			start_with_trial - skip certain trials from the grid search
			run_after_each_execution - lambda function to run after each execution with the model
			**kwargs - for model.fit()
		"""
		
		# create and init hyperparameters
		start_with_trial = start_with_trial - 1 # trial #1 is 0
		hp = Hyperparameters(start_with_trial)
		_ = self.build_model(hp)
		
		# start grid search
		max_trials = hp.trials()
		for trial in range(start_with_trial, max_trials):
			print('Trial %d of %d' % (trial+1, max_trials))
			hyperparameters = hp.getAll()
			print('Hyperparameters:', '{%s}' % ', '.join('%s: %s' % p for p in hyperparameters.items()))
			
			# update hyperparameters
			if 'batch_size' in hp: kwargs['batch_size'] = hp.get('batch_size')
			if 'epochs' in hp: kwargs['epochs'] = hp.get('epochs')
			
			# multiple executions
			execution_logs = list()
			predictions = list()
			best_score = None
			for execution in range(self.executions_per_trial):
				# set random seed
				seed = int.from_bytes(urandom(8), 'little')
				tf.random.set_seed(seed)
				
				# build model
				model = self.build_model(hp)
				
				# fit model
				history = model.fit(data, **kwargs)
				history = history.history
				
				# evaluate model
				execution_logs.append({
					'seed': seed,
					'objective': history[self.objective],
					'metrics': history,
				})
				if self.output_predictions:
					preds = {'train': model.predict(data).logits.tolist()}
					if 'validation_data' in kwargs: preds['val'] = model.predict(kwargs['validation_data']).logits.tolist()
					predictions.append(preds)
				
				# save model
				current_score = history[self.objective][-1]
				if best_score is None or self.extremum(best_score, current_score) == current_score:
					best_score = current_score
					if self.save_best_model_weights:
						makedirs(self.output_dir, exist_ok=True)
						model.save_weights(join(self.output_dir, 'best-model.h5'))
				
				# run after each execution
				if run_after_each_execution is not None:
					run_after_each_execution(model)
			
			# evaluate trial
			best_execution = self.extremum(enumerate(execution_logs), key=lambda x: x[1]['objective'][-1])[0]
			self.log.append({
				'hyperparameters': hyperparameters,
				'best_execution': best_execution,
				'executions': execution_logs,
			})
			self.save_log()
			self.save_best_configuration()
			if self.output_predictions:
				self.predictions.append(predictions)
				self.save_predictions()
			print('Best result:', ' - '.join('%s: %.4f' % (m, s[-1]) for m, s in execution_logs[best_execution]['metrics'].items()))
			print()
			
			# next trial
			hp.nextTrial()
		
		# print best hyperparameters
		best_configuration = self.get_best_configuration()
		print('Best hyperparameters:', '{%s}' % ', '.join('%s: %s' % p for p in best_configuration['hyperparameters'].items()))
		print('Best result:', ' - '.join('%s: %.4f' % (m, s[-1]) for m, s in best_configuration['metrics'].items()))
	
	def get_best_configuration(self):
		best_trial = self.extremum(self.log, key=lambda x: x['executions'][x['best_execution']]['objective'][-1])
		best_execution = best_trial['executions'][best_trial['best_execution']]
		return {
			'hyperparameters': best_trial['hyperparameters'],
			'seed': best_execution['seed'],
			'objective': best_execution['objective'],
			'metrics': best_execution['metrics'],
		}
	
	def _save(self, filename, data):
		if self.output_dir is None: return
		makedirs(self.output_dir, exist_ok=True)
		with open(join(self.output_dir, filename), 'w', encoding='UTF-8') as file:
			json.dump(data, file, indent='  ', ensure_ascii=False)
	
	def save_log(self):
		self._save('log.json', self.log)
	
	def save_best_configuration(self):
		self._save('best-configuration.json', self.get_best_configuration())
	
	def save_predictions(self):
		self._save('predictions.json', self.predictions)

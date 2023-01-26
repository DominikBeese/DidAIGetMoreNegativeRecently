import tensorflow as tf
from tensorflow_addons.utils.types import FloatTensorLike

class SlantedTriangularLearningRate(tf.keras.optimizers.schedules.LearningRateSchedule):
	""" https://www.aclweb.org/anthology/P18-1031/ """

	def __init__(
		self,
		maximum_learning_rate: FloatTensorLike,
		number_of_iterations: FloatTensorLike,
		cut_frac: FloatTensorLike = 0.1,
		ratio: FloatTensorLike = 32,
		name: str = None
	):
		super(SlantedTriangularLearningRate, self).__init__()
		self.maximum_learning_rate = maximum_learning_rate
		self.number_of_iterations = number_of_iterations
		self.cut_frac = cut_frac
		self.ratio = ratio
		self.name = name
	
	def __call__(self, step):
		with tf.name_scope(self.name or 'SlantedTriangularLearningRate') as name:
			lr_max = tf.convert_to_tensor(
				self.maximum_learning_rate, name='maximum_learning_rate'
			)
			dtype = lr_max.dtype
			cut_frac = tf.cast(self.cut_frac, dtype)
			ratio = tf.cast(self.ratio, dtype)
			T = tf.cast(self.number_of_iterations, dtype)
			t = tf.cast(step, dtype)
			cut = tf.math.floor(T * cut_frac)
			p = tf.cond(
				t < cut,
				lambda: t / cut,
				lambda: 1 - (t - cut) / (cut * (1 / cut_frac - 1))
			)
			return lr_max * (1 + p * (ratio - 1)) / ratio
	
	def get_config(self):
		return {
			'maximum_learning_rate': self.maximum_learning_rate,
			'number_of_iterations': self.number_of_iterations,
			'cut_frac': self.cut_frac,
			'ratio': self.ratio,
			'name': self.name,
		}

class ThreeClassAccuracy(tf.keras.metrics.Accuracy):
	def update_state(self, y_true, y_pred, sample_weight=None):
		super(ThreeClassAccuracy, self).update_state(
			tf.where(y_true <= -0.1, -1, tf.where(y_true >= 0.1, 1, 0)),
			tf.where(y_pred <= -0.1, -1, tf.where(y_pred >= 0.1, 1, 0)),
			sample_weight
		)

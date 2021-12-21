import numpy as np

from sklearn.datasets import fetch_openml

class MakeDataset():

	@classmethod
	def load_data(cls):
		np.random.seed(42)
		return fetch_openml("titanic", version=1, as_frame=True, return_X_y=True)


from sklearn.datasets import fetch_openml

class MakeDataset():

	@classmethod
	def load_data(cls):
		"""Get Kaggle's Titanic dataset
		
		Returns:
		    Tuple: The Titanic dataset
		"""
		return fetch_openml("titanic", version=1, as_frame=True, return_X_y=True)

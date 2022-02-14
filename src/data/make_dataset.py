import pandas as pd

from sklearn.datasets import fetch_openml

class MakeDataset():

	@classmethod
	def load_training_data(cls):
		"""Get Kaggle's Titanic training dataset
		
		Returns:
		    Tuple: The Titanic training dataset using fetch_openml
		"""
		X, y = fetch_openml("titanic", version=1, as_frame=True, return_X_y=True)
		X.columns = X.columns.str.lower()
		return (X, y)


	@classmethod
	def load_training_data2(cls):
		"""Get Kaggle's Titanic training dataset from downloaded files
		
		Returns:
		    Tuple: The Titanic training dataset
		"""
		X = pd.read_csv('../data/raw/train.csv')
		y = pd.read_csv('../data/raw/gender_submission.csv')
		return (X, y)

	@classmethod
	def load_testing_data(cls):
		"""Get Kaggle's Titanic testing dataset from downloaded file
		
		Returns:
		    DataFrame: The Titanic testing dataset
		"""
		test = pd.read_csv('../data/raw/test.csv')
		test.columns = test.columns.str.lower()
		return test

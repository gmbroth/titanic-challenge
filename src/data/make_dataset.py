
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV

# A Python module is simply a Python source file, which can expose classes, functions and global variables.
# A Python package is simply a directory of Python module(s).
# So PEP 8 tells you that:
# packages (directories) should have short, all-lowercase names, preferably without underscores;
# modules (filenames) should have short, all-lowercase names, and they can contain underscores;
# classes should use the CapWords convention.

class MakeDataset():

	@classmethod
	def load_data(cls):
		np.random.seed(42)
		X, y = fetch_openml("titanic", version=1, as_frame=True, return_X_y=True)
		# Columns to drop because not predictive:
		# boat - Lifeboat (if survived)
		# body - Body number (if did not survive and body was recovered)
		# home.dest - home/destination
		# Columns to drop because high percentage of null values: cabin
		X.drop(['boat', 'body', 'cabin', 'home.dest'], axis=1, inplace=True)
		return train_test_split(X, y, stratify=y, test_size=0.2)	

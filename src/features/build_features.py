
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

class BuildFeatures():

	def build_features(self, X, y):

		# Columns being dropped because they're probably not predictive:
		# 	boat - Lifeboat (if survived)
		# 	body - Body number (if did not survive and body was recovered)
		# 	home.dest - home/destination
		# Columns to drop because contains a high percentage of null values: cabin

		X.drop(['boat', 'body', 'cabin', 'home.dest'], axis=1, inplace=True)
		X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2)
		pd.set_option('mode.chained_assignment', None)  # fuck the pandas slice vs copy "solution"
		for dataset in [X_train, X_test]:
		    dataset['family_size'] = dataset['parch'] + dataset['sibsp']
		    dataset.drop(['parch', 'sibsp'], axis=1, inplace=True)
		    dataset['is_alone'] = 1
		    dataset['is_alone'].loc[dataset['family_size'] > 1] = 0

		return X_train, X_test, y_train, y_test


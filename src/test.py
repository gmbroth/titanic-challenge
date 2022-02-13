import logging
import numpy as np
import os
import pandas as pd
import sys
import unittest

from data.make_dataset import MakeDataset
from features.build_features import BuildFeatures
from logging.config import fileConfig
from models.train_model import TrainModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from visualization.visualize import Plot


class Test(unittest.TestCase):

	@classmethod
	def setUpClass(cls):
		np.random.seed(42)


	@unittest.skipIf(False, "Skip test_classifier")
	def test_classifier(self):
		log = logging.getLogger()
		"""      
		Please see https://jaketae.github.io/study/sklearn-pipeline/ on which my 
		implementation is based.
		"""
		X, y = MakeDataset.load_training_data();
		self.assertTrue(not (X.empty or y.empty), f'expected training data, found none')

		test = MakeDataset.load_testing_data();
		self.assertTrue(not test.empty and test.shape[0] == 418, f'expected valid testing data')

		features = BuildFeatures(X, y, test)
		X_train, X_test, y_train, y_test, test = features.engineer();
		self.assertTrue(not (X_train.empty or X_test.empty or y_train.empty or y_test.empty), 
			f'missing expected data')

		categorical_columns = ['embarked', 'sex', 'pclass', 'title', 'is_alone']
		numerical_columns   = ['age', 'fare', 'family_size']
		preprocessor = TrainModel.transformerFor(categorical_columns, numerical_columns)
		self.assertTrue(preprocessor and preprocessor.transformers, f'missing expected preprocessor')
		self.assertTrue(len(preprocessor.transformers) == 2, f'wrong number of transformers')

		classifier = RandomForestClassifier()
		pipeline = TrainModel.pipelineFor(preprocessor, classifier)
		self.assertTrue(pipeline, f'missing expected pipeline')

		parameters = TrainModel.tunedParameters()
		random_search = RandomizedSearchCV(pipeline,
			param_distributions=parameters, n_iter=100)
		self.assertTrue(random_search, f'missing expected search result')

		random_search.fit(X_train, y_train)
		y_pred = random_search.predict(X_test)

		log.info(f'cross_val_score = {cross_val_score(pipeline, X_train, y_train, cv=5, scoring="accuracy").mean()}\n')
		log.info(f'optimized score = {random_search.best_score_}\n')
		log.info(f'predictions = {y_pred[:5]} versus test =\n{y_test[:5]}\n')
		log.info(f'Classification report:\n{classification_report(y_test, y_pred)}\n')
		if log.isEnabledFor(logging.INFO):
			Plot.plot_confusion_matrix(random_search, X_test, y_test)

		# Let's make a prediction on Kaggle's submission test data ...
		Y_pred = random_search.predict(test.drop(['passengerid',], axis=1, inplace=False, errors='ignore'))
		# ... and write the results to file in the prescribed format:
		submission = pd.DataFrame({
		        "PassengerId": test['passengerid'],
		        "Survived":    Y_pred
		    })
		self.assertTrue(submission.shape[0] == 418, f'expected 418 rows, instead found {submission.shape[0]}')
		submission.to_csv('../models/submission.csv', index=False)


if __name__ == '__main__':
	fileConfig('../logging_config.ini')
	try:
	    sys.exit(unittest.main())
	except Exception:
	    logging.exception("Exception in unittest.main(): ")
	    sys.exit(1)


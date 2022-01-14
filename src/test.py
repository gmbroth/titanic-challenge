import unittest
from data.make_dataset import MakeDataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from features.build_features import BuildFeatures
from models.train_model import TrainModel
from visualization.visualize import Plot
import numpy as np

class Test(unittest.TestCase):

	@classmethod
	def setUpClass(cls):
		np.random.seed(42)


	@unittest.skipIf(False, "Skip test_classifier")
	def test_classifier(self):
		X, y = MakeDataset.load_data();
		self.assertTrue(not (X.empty or y.empty), f'expected data, found none')

		X_train, X_test, y_train, y_test = BuildFeatures(X, y).engineerFeatures();
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
		print(f'cross_val_score = {cross_val_score(pipeline, X_train, y_train, cv=5, scoring="accuracy").mean()}\n')
		print(f'optimized score = {random_search.best_score_}\n')
		# print(f'optimized parameters = {random_search.best_params_}\n')

		print(f'predictions = {y_pred[:5]} versus test =\n{y_test[:5]}\n')
		print(f'Classification report:\n{classification_report(y_test, y_pred)}\n')

		Plot.plot_confusion_matrix(random_search, X_test, y_test)


if __name__ == '__main__':
	print(f'main')
	unittest.main()

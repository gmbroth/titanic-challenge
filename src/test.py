import unittest
from data.make_dataset import MakeDataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from features.build_features import BuildFeatures
from models.train_model import TrainModel

class Test(unittest.TestCase):

	def test(self):
		X, y = MakeDataset.load_data();
		self.assertTrue(X.size == 17017, f'expected 17017 columns, found {X.size}')

		X_train, X_test, y_train, y_test = BuildFeatures(X, y).engineerFeatures();
		self.assertTrue(not (X_train.empty or X_test.empty or y_train.empty or y_test.empty), f'missing expected dataframes')

		categorical_columns = ['embarked', 'sex', 'pclass', 'title', 'is_alone']
		numerical_columns   = ['age', 'fare', 'family_size']
		preprocessor = TrainModel.transformerFor(categorical_columns, numerical_columns)
		self.assertTrue(preprocessor and preprocessor.transformers, f'missing expected preprocessor')
		self.assertTrue(len(preprocessor.transformers) == 2, f'wrong number of transformers')

		classifier = RandomForestClassifier()
		pipeline = TrainModel.pipelineFor(preprocessor, classifier)
		self.assertTrue(pipeline, f'missing expected pipeline')

		print(f'cross_val_score = {cross_val_score(pipeline, X_train, y_train, cv=5, scoring="accuracy").mean()}')


if __name__ == '__main__':
	unittest.main()


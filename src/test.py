import unittest
from data.make_dataset import MakeDataset
from features.build_features import BuildFeatures

class Test(unittest.TestCase):

	def test(self):
		X, y = MakeDataset.load_data();
		self.assertTrue(X.size == 17017, f'expected 17017 columns, found {X.size}')
		X_train, X_test, y_train, y_test = BuildFeatures().build_features(X, y);
		self.assertTrue(X_train.shape[1] == 9, f'expected 9 columns, found {X_train.shape[1]}')

if __name__ == '__main__':
	unittest.main()


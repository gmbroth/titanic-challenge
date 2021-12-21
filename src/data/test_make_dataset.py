import unittest
from make_dataset import MakeDataset

class TestLoadData(unittest.TestCase):

	def test_load_data(self):
		X, y = MakeDataset.load_data();
		self.assertTrue(X.size == 17017, f'expected 17017 columns, found {X.size}')

if __name__ == '__main__':
	unittest.main()


import unittest
from make_dataset import MakeDataset

# A Python module is simply a Python source file, which can expose classes, functions and global variables.
# A Python package is simply a directory of Python module(s).
# So PEP 8 tells you that:
# packages (directories) should have short, all-lowercase names, preferably without underscores;
# modules (filenames) should have short, all-lowercase names, and they can contain underscores;
# classes should use the CapWords convention.

class TestLoadFeatures(unittest.TestCase):

	def test_load_data(self):
		X_train, X_test, y_train, y_test = MakeDataset.load_data();
		self.assertTrue(X_train.columns.size == 9, f'expected 9 columns, found {X_train.columns.size}')

if __name__ == '__main__':
	unittest.main()


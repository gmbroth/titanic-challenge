
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder

class TrainModel():

	@classmethod
	def transformerFor(cls, cat_cols, num_cols):
		"""Construct a column transformer for the named columns

		Please see https://jaketae.github.io/study/sklearn-pipeline/ on 
		which this implementation is based.
		
		Args:
		    cat_cols (List): Categorical column names
		    num_cols (List): Numerical column names
		
		Returns:
		    ColumnTransformer: a column transformer
		"""
		# Categorical column transformer
		cat_transformer = Pipeline(steps=[
		    ('imputer', SimpleImputer(strategy='most_frequent')),
		    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False)),
		    ('pca', PCA(n_components=10))
		])

		# Numerical column transformer
		num_transformer = Pipeline(steps=[
		    ('imputer', KNNImputer(n_neighbors=5)),
		    ('scaler', RobustScaler())
		])

		return ColumnTransformer(
		    transformers=[
		        ('num', num_transformer, num_cols),
		        ('cat', cat_transformer, cat_cols)
		    ])		
		

	@classmethod
	def pipelineFor(cls, preprocessor, classifier):
		"""Construct a pipeline for the specified preprocessor and classifier
		
		Args:
		    preprocessor (ColumnTransformer): A column transformer
		    classifier (Classifier): A model classifier
		
		Returns:
		    Pipeline: A Pipeline suitable for classification use
		"""
		return Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', classifier)])


	@classmethod
	def tunedParameters(cls):
		"""Define search parameters
		
		Returns:
		    Dictionary: A dictionary of key-value search parameters
		"""
		num_transformer_dist = {'preprocessor__num__imputer__n_neighbors': list(range(2, 15)),
		                        'preprocessor__num__imputer__add_indicator': [True, False]}

		cat_transformer_dist = {'preprocessor__cat__imputer__strategy': ['most_frequent', 'constant'],
		                        'preprocessor__cat__imputer__add_indicator': [True, False],
		                        'preprocessor__cat__pca__n_components': list(range(2, 15))}

		random_forest_dist = {'classifier__n_estimators': list(range(50, 500)),
		                      'classifier__max_depth': list(range(2, 20)),
		                      'classifier__bootstrap': [True, False]}

		return {**num_transformer_dist, **cat_transformer_dist, **random_forest_dist}



import pandas as pd

from sklearn.model_selection import train_test_split

class BuildFeatures():
    
    def __init__(self, X, y, test):
        self.X, self.y, self.test = (X, y, test)  


    def engineer(self):
        """Engineer features of the configured data
        
        Returns:
            Tuple: The engineered features
        """

        # The feature engineering performed in this method is nicked from the incredibly
        # useful (and lucid) post https://jaketae.github.io/study/sklearn-pipeline/

        # 1. Columns being dropped because they're probably not predictive:
        #   boat - Lifeboat (if survived)
        #   body - Body number (if did not survive and body was recovered)
        #   home.dest - home/destination
        # 2. Column dropped because contains a high percentage of null values: cabin
        for dataset in [self.X, self.test]:
            dataset.drop(['boat', 'body', 'cabin', 'home.dest'], axis=1, inplace=True, errors='ignore')

        # Split data for train and test.
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, stratify=self.y, test_size=0.2)
        pd.set_option('mode.chained_assignment', None)  # phuck pandas and this so-called slice vs copy "solution"
        for dataset in [X_train, X_test, self.test]:
            dataset['family_size'] = dataset['parch'] + dataset['sibsp']
            dataset.drop(['parch', 'sibsp'], axis=1, inplace=True, errors='ignore')
            dataset['is_alone'] = 1
            dataset['is_alone'].loc[dataset['family_size'] > 1] = 0
            dataset['title'] =  dataset['name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
            dataset.drop(['name', 'ticket'], axis=1, inplace=True, errors='ignore')

            # Combine some of the many titles in the data.
            pd.crosstab(X_train['title'], X_train['sex'])
            X_comb = pd.concat([X_train, X_test, self.test]) 
            # Identify rarely occurring titles: Major, Capt, Rev, etc.
            rare_titles = (X_comb['title'].value_counts() < 10)
            # Treat both the titles "Miss" and "Mrs" as simply "Mrs"
            dataset.title.loc[dataset.title == 'Miss'] = 'Mrs'
            dataset['title'] = dataset.title.apply(lambda x: 'rare' if rare_titles[x] else x)

        return X_train, X_test, y_train, y_test, self.test

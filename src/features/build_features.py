
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

class BuildFeatures():
    
    def __init__(self, X, y):
        self.X, self.y = (X, y)  


    def engineerFeatures(self):
        """Summary
        
        Returns:
            TYPE: Description
        """
        # The feature engineering performed below is nicked from https://jaketae.github.io/study/sklearn-pipeline/

        # Columns being dropped because they're probably not predictive:
        #   boat - Lifeboat (if survived)
        #   body - Body number (if did not survive and body was recovered)
        #   home.dest - home/destination
        # Columns to drop because contains a high percentage of null values: cabin
        self.X.drop(['boat', 'body', 'cabin', 'home.dest'], axis=1, inplace=True)
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, stratify=self.y, test_size=0.2)
        pd.set_option('mode.chained_assignment', None)  # fuck pandas and this so-called slice vs copy "solution"
        for dataset in [X_train, X_test]:
            dataset['family_size'] = dataset['parch'] + dataset['sibsp']
            dataset.drop(['parch', 'sibsp'], axis=1, inplace=True)
            dataset['is_alone'] = 1
            dataset['is_alone'].loc[dataset['family_size'] > 1] = 0
            dataset['title'] =  dataset['name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
            dataset.drop(['name', 'ticket'], axis=1, inplace=True)

            # Combine some of the many titles in the data.
            pd.crosstab(X_train['title'], X_train['sex'])
            X_comb = pd.concat([X_train, X_test])
            # Identify rarely occurring titles: Major, Capt, Rev, etc.
            rare_titles = (X_comb['title'].value_counts() < 10)
            # Treat both the titles "Miss" and "Mrs" as simply "Mrs."
            dataset.title.loc[dataset.title == 'Miss'] = 'Mrs'
            dataset['title'] = dataset.title.apply(lambda x: 'rare' if rare_titles[x] else x)

        return X_train, X_test, y_train, y_test



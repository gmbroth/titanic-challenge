import matplotlib.pyplot as plt

from sklearn.metrics import plot_confusion_matrix
 
class Plot():

    @classmethod
    def plot_confusion_matrix(cls, model_selection, X_test, y_test):
        """Plot a confusion matrix
        
        Args:
            model_selection (TYPE): A model selection object
            X_test (DataFrame): Test data
            y_test (DataFrame): Test results
        """
        plot_confusion_matrix(model_selection, X_test, y_test, cmap=plt.cm.Blues, normalize='true')
        plt.show()  
 

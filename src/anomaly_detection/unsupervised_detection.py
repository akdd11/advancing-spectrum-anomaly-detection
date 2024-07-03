import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import skew
import warnings


from sklearn.exceptions import NotFittedError
from sklearn.neighbors import LocalOutlierFactor

# Implemented methods for unsupervised anomaly detection
unsupervised_methods = ['energy_detector', 'lof']


def sigmoid(x):
    """Sigmoid function.
    
    x: float or array_like
        Argument of the sigmoid function.
    """
    return 1 / (1 + np.exp(-x))


class EnergyDetector:
    """Threshold-based classifier for anomaly detection inspired by energy detector from signal processing.
    
    It uses the a percentile of the mean or median difference between original data and the
    digital twin data as a threshold. If the difference is bigger than the threshold, the
    measurement is classified as an anomaly.

    The threshold is obtained from the training data.
    """

    def __init__(self, percentile=0.9, probability=False, statistic='mean', use_absolute_value=False):
        """Initialize the classifier.
        
        Properties:
        percentile : float
            Percentile to use for the threshold (0-1).
        probability : bool
            If True, the probability of the measurement being an anomaly is returned instead of
            the binary classification. 
        statistic : str
            Statistic to use for the threshold. Can be 'mean' or 'median'.
        use_absolute_value : bool
            If True, the absolute value of the statistic is used.
        """

        self.percentile = percentile*100
        self.probability = probability

        if statistic in ['mean', 'median', 'skewness', 'var']:
            self.statisitc = statistic
        else:
            raise NotImplementedError(f'Statistic {statistic} is not implemented.^'\
                                      ' Choose from "mean", "median", "skewness", "var".')
        
        self.use_absolute_value = use_absolute_value
        self.threshold = np.nan

    def fit(self, X_train):
        """Train the classifier.
        
        Parameters:
        X_train : numpy.ndarray
            Training data.
        """

        if self.use_absolute_value:
            X_train = np.abs(X_train)
        
        if self.statisitc == 'mean':
            X_train = np.mean(X_train, axis=1)
        elif self.statisitc == 'median':
            X_train = np.median(X_train, axis=1)
        elif self.statisitc == 'skewness':
            X_train = skew(X_train,axis=1)
        elif self.statisitc == 'var':
            X_train = np.var(X_train,axis=1)

        self.threshold = np.percentile(X_train, self.percentile)

    def predict(self, X_test):
        """Predict anomalies.
        
        Parameters:
        X_test : numpy.ndarray
            Test data.
        """

        if np.isnan(self.threshold):
            raise NotFittedError('The classifier has to be fitted first.')

        if self.use_absolute_value:
            X_test = np.abs(X_test)
        
        if self.statisitc == 'mean':
            X_test = np.mean(X_test, axis=1)
        elif self.statisitc == 'median':
            X_test = np.median(X_test, axis=1)
        elif self.statisitc == 'skewness':
            X_test = skew(X_test,axis=1)
        elif self.statisitc == 'var':
            X_test = np.var(X_test,axis=1)

        if self.probability:
            return sigmoid(X_test - self.threshold)
        else:
            return X_test > self.threshold
        

    def visualize_distributions(self, X_train, X_test, y_train, y_test):
        """Visualize the distributions of the training and test data.
        
        Parameters:
        X_train : numpy.ndarray
            Training data.
        X_test : numpy.ndarray
            Test data.
        y_train : numpy.ndarray
            Training labels.
        y_test : numpy.ndarray
            Test labels.
        """

        if self.use_absolute_value:
            X_train = np.abs(X_train)
            X_test = np.abs(X_test)
        
        if self.statisitc == 'mean':
            X_train = np.mean(X_train, axis=1)
            X_test = np.mean(X_test, axis=1)
        elif self.statisitc == 'median':
            X_train = np.median(X_train, axis=1)
            X_test = np.median(X_test, axis=1)

        # Plot the distributions
        sns.kdeplot(X_train[y_train == 0], color='C0', linestyle='-')
        sns.kdeplot(X_train[y_train == 1], color='C0', linestyle='--')
        sns.kdeplot(X_test[y_test == 0], color='C1', linestyle='-')
        sns.kdeplot(X_test[y_test == 1], color='C1', linestyle='--')

        # Create the legend entries
        plt.plot(np.nan, np.nan, color='black', linestyle='-', label='Normal')
        plt.plot(np.nan, np.nan, color='black', linestyle='--', label='Anomaly')
        plt.plot(np.nan, np.nan, color='C0', linestyle='-', label='Train')
        plt.plot(np.nan, np.nan, color='C1', linestyle='-', label='Test')

        # Plot the threshold
        if np.isnan(self.threshold):
            warnings.warn('The classifier has not yet been fitted. Hence, no threshold is shown.')
        else:
            plt.axvline(self.threshold, color='red', linestyle=':', label='Threshold')


        plt.xlabel('Measurement difference (original - digital twin) [dB]')
        plt.legend()
        plt.show()


class LocalOutlierFactorLearning:
    """LocalOutlierFactorLearing

     It measures the local deviation of the density of a given sample with respect to its neighbors.
     Based on this, it calculates an anomaly score, the Local Outlier Factor (LOF), for each sample.
    """

    def __init__(self, n_neighbors=20, algorithm='auto', contamination='auto', probability=False):
        """Initialization

            n_neighbors : int
                no of neighbors (nearest) to consider for density calculation.
            alogrithm : {auto,ball_tree,kd_tree, brute}
                algorithm to use for calculating nearest neighbors.
            probability : bool
                If True, the probability of the measurement being an anomaly is returned instead of
                the binary classification.
        """

        self.n_neighbors = n_neighbors
        self.algorithm = algorithm
        self.contamination = contamination
        self.probability = probability

        self.cls = LocalOutlierFactor(n_neighbors=self.n_neighbors,
                                      algorithm=self.algorithm,
                                      contamination=self.contamination,
                                      novelty=True)


    def fit(self, X_train):
        """Train the model

        Parameter:
        X_train : np.ndarray
            Training data
        """  

        self.cls.fit(X_train)

    def predict(self, X_test):
        """Make anamoly predictions

        Parameter
        X_test : np.ndarray
            Test data
        """
        if self.probability:
            # a minus sign is added, as the decision function return bigger values for inliers
            return sigmoid(-self.cls.decision_function(X_test))
        else:
            return self.cls.predict(X_test)

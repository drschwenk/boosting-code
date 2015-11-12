import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import clone
import math

class AdaBoostBinaryClassifier(object):
    '''
    INPUT:
    - n_estimator (int)
      * The number of estimators to use in boosting
      * Default: 50

    - learning_rate (float)
      * Determines how fast the error would shrink
      * Lower learning rate means more accurate decision boundary,
        but slower to converge
      * Default: 1
    '''

    def __init__(self,
                 n_estimators=50,
                 learning_rate=1):

        self.base_estimator = DecisionTreeClassifier(max_depth=1)
        self.n_estimator = n_estimators
        self.learning_rate = learning_rate

        # Will be filled-in in the fit() step
        self.estimators_ = []
        self.estimator_weight_ = np.zeros(self.n_estimator, dtype=np.float)

    def fit(self, x, y):
        '''
        INPUT:
        - x: 2d numpy array, feature matrix
        - y: numpy array, labels

        Build the estimators for the AdaBoost estimator.
        '''

        sample_weight = np.ones(len(y), dtype=np.float)/len(y)
        for idx in xrange(self.n_estimator):
            tree, sample_weight, estimator_weight = self._boost(x, y, sample_weight)
            self.estimator_weight_[idx] = estimator_weight
            self.estimators_.append(tree)
        return None

    def _boost(self, x, y, sample_weight):
        '''
        INPUT:
        - x: 2d numpy array, feature matrix
        - y: numpy array, labels
        - sample_weight: numpy array

        OUTPUT:
        - estimator: DecisionTreeClassifier
        - sample_weight: numpy array (updated weights)
        - estimator_weight: float (weight of estimator)

        Go through one iteration of the AdaBoost algorithm. Build one estimator.
        '''

        estimator = clone(self.base_estimator)

        # step a
        estimator.fit(x,y, sample_weight = sample_weight)
        y_hat = estimator.predict(x)

        # step b
        err = sample_weight.dot((y_hat != y) + 0) / np.sum(sample_weight)

        # step c
        estimator_weight = np.log((1 - err) / err)

        # step d
        sample_weight = sample_weight * np.exp(estimator_weight *\
                        ((y_hat != y) + 0))

        return estimator, sample_weight, estimator_weight


    def predict(self, x):
        '''
        INPUT:
        - x: 2d numpy array, feature matrix

        OUTPUT:
        - labels: numpy array of predictions (0 or 1)
        '''
        G_m = np.array([tree.predict(x) for tree in self.estimators_])
        G = (self.estimator_weight_.dot(G_m) > 0) + 0
        return G

    def score(self, x, y):
        '''
        INPUT:
        - x: 2d numpy array, feature matrix
        - y: numpy array, labels

        OUTPUT:
        - score: float (accuracy score between 0 and 1)
        '''

        y_hat = self.predict(x)
        return np.sum(y_hat == y)/float(len(y))


from sklearn.cross_validation import train_test_split
data = np.genfromtxt('./data/spam.csv', delimiter=',')

y = data[:, -1]
x = data[:, 0:-1]

train_x, test_x, train_y, test_y = train_test_split(x, y)

my_ada = AdaBoostBinaryClassifier(n_estimators=50)
my_ada.fit(train_x, train_y)
my_ada.score(test_x, test_y)
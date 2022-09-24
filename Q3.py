from sklearn.datasets import load_iris
import numpy as np
from numpy import exp, pi, sqrt, log, e

class NBC ():
    def __init__(self,feature_types,num_classes):
        self.feature_types = feature_types
        self.num_classes = num_classes
    
    def prepwork(self, Xtrain,ytrain, epsilon=1e-6):
        self.label = np.unique(ytrain)  
        self.epsilon = epsilon
        n_features = len(self.feature_types)
        n_classes = len(self.label)
        self.xmean = np.zeros((n_classes, n_features))  
        self.xvar = np.zeros((n_classes, n_features))
        self.classcount = np.zeros(n_classes, dtype=np.float64)
        self.yprior = np.zeros(len(self.label), dtype=np.float64)

    
    def fit(self, Xtrain, ytrain):
        self. prepwork(Xtrain,ytrain)
        for i, y_i in enumerate(self.label): 
            X_i = Xtrain[ytrain == y_i, :]
            self.xmean[i, :] = np.mean(X_i, axis=0)  
            self.xvar[i, :] = np.var(X_i, axis=0)
            self.classcount[i] += X_i.shape[0]          
            self.yprior = self.classcount / self.classcount.sum()
        return self.xmean,self.xvar,self.yprior
    
    def likelihood(self, Xtest):

        likelihood = []
        for i in range(len(self.label)):
            gdis= (1 / sqrt(2 * pi * self.xvar[i, :]) * exp(-1 * (Xtest - self.xmean[i, :]) ** 2 / (2 * self.xvar[i, :])))+ self.epsilon
            loggdis = log(gdis).sum(axis=1)
            logy = log(self.yprior[i])
            likelihood.append(logy + loggdis) 
        likelihood = np.array(likelihood).T 
        return likelihood    
    
    def prediction (self, Xtest):
        prediction = []
        for label in range (len (Xtest)):
            labelpredict= np.argmax(self.likelihood(Xtest)[label])
            prediction.append(labelpredict)
        return prediction
    
    
iris = load_iris()
X, y = iris['data'], iris['target']
N, D = X.shape
Ntrain = int(0.9 * N)
shuffler = np.random.permutation(N)
Xtrain = X[shuffler[:Ntrain]]
ytrain = y[shuffler[:Ntrain]]
Xtest = X[shuffler[Ntrain:]]
ytest = y[shuffler[Ntrain:]]

nbc=NBC(feature_types=['r', 'r', 'r', 'r'],num_classes=3)
nbc.fit(Xtrain, ytrain)
yhat = nbc.prediction(Xtest)
test_accuracy = np.mean(yhat == ytest)

print ("The accuracy is %.2f%%!" % (test_accuracy * 100))
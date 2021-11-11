import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

import warnings;

warnings.filterwarnings('ignore')
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score, StratifiedKFold

skf = StratifiedKFold(n_splits=10)

names = ['Sample code number', 'Clump Thickness',
         'Uniformity of Cell Size', 'Uniformity of Cell Shape',
         'Marginal Adhesion', 'Single Epithelial CellSize',
         'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli',
         'Mitoses', 'Class']

data = pd.read_csv('breast-cancer-wisconsin.data', sep=',', names=names)

# drop missing value
data = data[data['Bare Nuclei'] != "?"]

X = data.drop(columns='Class')
Y = data['Class']

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2, stratify=Y, shuffle=True, random_state=1)


# Mainboard of the program, passes the data to scaling functions, and modeling functions
class AutoProcess():
    def __init__(self, verbose=False):
        super(AutoProcess, self).__init__()
        self.pp = Preprocess
        self.verbose = verbose

    def run(self, X, Y, Xt, Yt):
        methods = []
        scores = []
        for num_process in ['standard', 'robust', 'minmax', 'maxabs']:
            if self.verbose:
                print("\n------------------------------------------------------\n")
                print(f"***Scaler : {num_process}")
            methods.append([num_process])
            pipeline = self.pp(num_process=num_process)
            X_processed, Xt_processed = pipeline.process(X, Xt)
            for model in ['logistic']:
                if self.verbose:
                    print(f"\nRegression model: {model}")
                doR = doRegression(model)
                doR.doR(X_processed, Y, Xt_processed, Yt)
            for model in ['dtree_entropy', 'dtree_gini', 'svc']:
                if self.verbose:
                    print(f"\nClassifier model: {model}")
                doC = doClassification(model)
                doC.doC(X_processed, Y, Xt_processed, Yt)
            print("*****************PCA*****************")
            for var in [.95, .99]:
                if self.verbose:
                    print(f"\n***PCA var: {var}")
                pcax, pcaxt = doPCA(var, X_processed, Xt_processed)
                for model in ['logistic']:
                    if self.verbose:
                        print(f"\nRegression model: {model}")
                    doR = doRegression(model)
                    doR.doR(pcax, Y, pcaxt, Yt)
                for model in ['dtree_entropy', 'dtree_gini', 'svc']:
                    if self.verbose:
                        print(f"\nClassifier model: {model}")
                    doC = doClassification(model)
                    doC.doC(pcax, Y, pcaxt, Yt)
        return


# Function that preprocess the data
class Preprocess():
    def __init__(self, num_process, verbose=False):
        super(Preprocess, self).__init__()
        self.num_process = num_process

        if num_process == 'standard':
            self.scaler = preprocessing.StandardScaler()
        elif num_process == 'minmax':
            self.scaler = preprocessing.MinMaxScaler()
        elif num_process == 'maxabs':
            self.scaler = preprocessing.MaxAbsScaler()
        elif num_process == 'robust':
            self.scaler = preprocessing.RobustScaler()
        else:
            raise ValueError("Supported 'num_process' : 'standard','minmax','maxabs','robust'")

        self.verbose = verbose

    def process(self, X, Xt):
        X_processed = self.scaler.fit_transform(X)
        Xt_processed = self.scaler.transform(Xt)

        return X_processed, Xt_processed


from sklearn.linear_model import LogisticRegression


# Function that builds a regression model
class doRegression():
    def __init__(self, model):
        super(doRegression, self).__init__()
        self.model = model

        if model == 'logistic':
            self.regression = LogisticRegression()
        else:
            raise ValueError("Supported 'model' : 'logistic'")

    def doR(self, x, y, xt, yt):
        self.regression.fit(x, y)

        print("Score:", round(self.regression.score(xt, yt) * 100, 2))


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

kfold = KFold(5, True, 1)


# Function that builds classification models
class doClassification():
    def __init__(self, model):
        super(doClassification, self).__init__()
        self.model = model

        if model == 'dtree_entropy':
            self.classifier = DecisionTreeClassifier(criterion='entropy')
        elif model == 'dtree_gini':
            self.classifier = DecisionTreeClassifier(criterion='gini')
        elif model == 'svc':
            self.classifier = SVC()
        else:
            raise ValueError("Supported 'model' :'dtree','svc'")

    def doC(self, x, y, xt, yt):
        self.classifier.fit(x, y)
        print("Score: ", round(self.classifier.score(xt, yt) * 100, 2))

        score = cross_val_score(self.classifier, xt, yt, cv=kfold, n_jobs=1, scoring='accuracy')
        print("Score with using kfold: ", round(np.mean(score) * 100, 2))


from sklearn.decomposition import PCA


# Transforms the data into PCA data
def doPCA(var, x, xt):
    pca = PCA(var)
    pca.fit(x)

    xpca = pca.transform(x)
    xtpca = pca.transform(xt)

    return xpca, xtpca


autoprocess = AutoProcess(verbose=True)
autoprocess.run(Xtrain, Ytrain, Xtest, Ytest)

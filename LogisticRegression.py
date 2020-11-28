import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.decomposition import PCA

df = pd.read_csv(r'C:\Users\meysam-sadat\PycharmProjects\diabetes.csv')
X = df.drop('Outcome',axis=1)
y = df.Outcome

sc = StandardScaler()
pca = PCA(n_components=8)
LR = LogisticRegression()
pipe = Pipeline(steps=[('StandardScaler', sc),('pca', pca),('logistic_Reg', LR)])
n_components = list(range(1, X.shape[1] + 1, 1))
C = np.logspace(-4, 4, 50)
penalty = ['l1', 'l2']
parameters = dict(pca__n_components= n_components, logistic_Reg__C = C,logistic_Reg__penalty = penalty)
clf_GS = GridSearchCV(pipe, parameters)
clf_GS.fit(X, y)
print('Best Penalty:', clf_GS.best_estimator_.get_params()['logistic_Reg__penalty'])
print('Best C:', clf_GS.best_estimator_.get_params()['logistic_Reg__C'])
print('Best Number Of Components:', clf_GS.best_estimator_.get_params()['pca__n_components'])
print(clf_GS.best_estimator_.get_params()['logistic_Reg'])
clf_GS.best_params_
pca = PCA(n_components=8)
X = pca.fit_transform(X)
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=142)
LR_optimal = LogisticRegression(penalty='l2',C=0.26)
LR_optimal.fit(x_train,y_train)
y_pred = LR_optimal.predict(x_test)
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.svm import SVR,LinearSVR
from sklearn.datasets import load_boston,load_diabetes
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

diabets = load_diabetes()
df = pd.DataFrame(diabets.data,columns=diabets.feature_names)
df['Target'] = diabets.target
df_corr = df.corr()
sns.heatmap(df_corr,annot=True)
# sns.distplot(df.Target)

print(diabets.DESCR)
df.shape
df_des = df.describe()
sns.pairplot(df.iloc[:,0:6])
sns.pairplot(df.iloc[:,6:11])

x = df.drop('Target',axis=1)
y = df['Target']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=142)

norm_list = [MinMaxScaler,StandardScaler]
n_com = [2,3,4,5,8,10]

r2_score_list = []
mse_list = []

for norm_items in norm_list:
    for n_c in n_com:
        my_norm = norm_items()
        x_train_l = my_norm.fit_transform(x_train)
        x_test_l = my_norm.transform(x_test)
        my_pca = PCA(n_c)
        x_train_l = my_pca.fit_transform(x_train_l)
        x_test_l = my_pca.transform(x_test_l)
        reg = LinearRegression()
        reg.fit(x_train_l,y_train)
        r2_score_list.append(r2_score(y_test,reg.predict(x_test_l)))
        mse_list.append(mean_squared_error(y_test,reg.predict(x_test_l)))

cnt = 0
for norm_items in norm_list:
    for n_c in n_com:
        print(f'{norm_items}: {n_c} : {r2_score_list[cnt]}: {mse_list[cnt]}')
sc = StandardScaler()
x_scaled = sc.fit_transform(x)

columns_pca = ['f1','f2','f3','f4','f5']
pca = PCA(n_components=5)
pca_x = pca.fit_transform(x_scaled)
pca_df = pd.DataFrame(pca_x,columns=columns_pca)
final_df = pca_df.join(y)
x_new = final_df.drop('Target',axis=1)

x_train,x_test,y_train,y_test = train_test_split(x_new,y,test_size=0.3,random_state=142)

reg = LinearRegression(n_jobs=-1)
reg.fit(x_train,y_train)
y_pred = reg.predict(x_test)
reg.score(x_test,y_test)
r2_score(y_test,y_pred)
mean_squared_error(y_test,y_pred)

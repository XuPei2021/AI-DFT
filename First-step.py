import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
from formula.data_process import train_dat_processing
from formula.dat_processing import feature_processing
import random
from sklearn.model_selection import train_test_split,GridSearchCV

train_data_path = 'file_path'
feature_path = 'file_path'
train_data,df_train,feature_columns = train_dat_processing(train_data_path)
# print(feature_columns)
for i in range(len(feature_columns)):
    globals()[feature_columns[i]] = train_data.T[i]
feature_names = feature_processing(feature_path)
d_1D_train = np.array([eval(token) for token in feature_names])
d_1D_train = pd.DataFrame(d_1D_train.T,columns=feature_names)
dict_all_y = df_train.iloc[:,1]
new_train = pd.concat([dict_all_y,d_1D_train],axis=1)

train_dim = 2  
train_score = []  
dim_index = []    
for i in range(100000):                    
    a = random.sample(feature_names, train_dim)
    b = [a]
    dim_index.append(b)
    a.append('f_m')
    train_data = new_train[a]
    x = train_data.iloc[:,0:train_dim+1]
    y = train_data.iloc[:,-1]
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=0)
    sc = StandardScaler()
    sc.fit(x)
    X_train_std = sc.transform(x_train)
    # X_test_std: object = sc.transform(x_test)
    model = SVC(kernel='rbf', C=1.0, random_state=42)
    # model = Lasso(alpha=0.01)
    model.fit(X_train_std,y_train)
    score = model.score(x_test, y_test)
    train_score.append(score)
number = train_score.index(max(train_score))
x_train = new_train[dim_index[10][0][:2]]
y_train = new_train.iloc[:,0]
sc = StandardScaler()
sc.fit(x_train)
X_train_std = sc.transform(x_train)
# X_test_std: object = sc.transform(x_test)
param = {'C':[1,5,10,20,30,40,50,60,70,80,90,100],
        'gamma':[0.1,0.2,0.3,0.4,0.5,1]}
model = SVC()
model = GridSearchCV(estimator=model,param_grid=param,n_jobs=-1,cv=10)

model.fit(x_train,y_train)
model.score(x_train,y_train)
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('./p/P1.csv')
data.head()
x = data.drop(columns=['material','p'])
y = data.iloc[:,2]
x_one = pd.get_dummies(x,columns=['Valence'])

x_valid = x_one.iloc[0:40,:]
y_valid = y[0:40]
x_train = x_one.iloc[40:,:]
y_train = y[40:]

model = RandomForestRegressor(n_estimators=50,max_depth=40)
model1 = RandomForestRegressor()
param = {'n_estimators':[1,5,10,50,100,500,1000,5000],
        'max_depth':[2,5,10,15,20,30],
        'max_features':[1,2,3,4,5,6,7],
        'min_samples_leaf':[2,5,10,15,20,30]}

model = GridSearchCV(estimator=model1,param_grid=param,n_jobs=-1,cv=10)
model.fit(x_train,y_train)
model.score(x_train,y_train)

import matplotlib
i = 70
# shap.initjs()
# print(x_valid.iloc[i,:],y_valid.values[i])
shap.force_plot(explainer.expected_value, shap_values[i,:], x_valid.iloc[i,:],matplotlib=True,figsize=(20,3), plot_cmap='"DrDb"')
# plt.savefig('./123.png')
shap.force_plot(explainer.expected_value, shap_values, x_valid,show=False)
shap.summary_plot(shap_values, x_valid,max_display=8,plot_size=0.5)
matplotlib = True
shap.plots.heatmap(shap_values1,max_display=11)
shap.plots.bar(shap_values1[70],show_data=True)
shap.plots.bar(shap_values1.cohorts(2).abs.mean(0))
expected_value = explainer.expected_value
shap.decision_plot(expected_value, shap_values[50:60], x_valid.columns)
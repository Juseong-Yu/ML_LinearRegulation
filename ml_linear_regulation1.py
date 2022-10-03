from sklearn import linear_model
from sklearn.metrics import accuracy_score 
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df_car = pd.DataFrame({'type':['A','B','C','D','E','F','G'],
'hp':[130,250,190,300,210,220,170],
'ef':[16.3,10.2,11.1,7.1,12.1,13.2,14.2]})

x = df_car['hp'].to_numpy()
x = x[:,np.newaxis]
y = df_car['ef'].to_numpy()

regr = linear_model.LinearRegression()
regr.fit(x,y)

df_car.plot(kind='scatter',x='hp',y='ef')
y_pred = regr.predict([[0],[300]])

plt.plot([0,300],y_pred)


print("계수: ",regr.coef_)
print("절편: ",regr.intercept_)
print('예측 점수 ')
new_car = round(regr.predict([[270]])[0],2)
print("270 마력 자동차의 예상 연비 : ",new_car,"km/l")
print("예측 점수 :",regr.score(x,y))
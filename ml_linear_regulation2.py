from sklearn import linear_model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df_car = pd.DataFrame({'type':['A','B','C','D','E','F','G'],
'hp':[130,250,190,300,210,220,170],
'ef':[16.3,10.2,11.1,7.1,12.1,13.2,14.2],
'weight':[1900,2600,2200,2900,2400,2300,2100]})

x = df_car[['hp','weight']]
y = df_car['ef']

regr = linear_model.LinearRegression()
regr.fit(x,y)
y_pred = regr.predict(x)

print("계수: ",regr.coef_)
print("절편: ",regr.intercept_)
print("예측 점수 :",regr.score(x,y))

new_car = round(regr.predict([[270,2500]])[0],2)
print("270 마력 2500kg 자동차의 예상 연비: ",new_car,"km/l")

sns.pairplot(df_car[['hp','ef','weight']])
sns.set(rc={'figure.figsize':(12,10)})
correlation_matrix = df_car.corr().round(2)
sns.heatmap(data=correlation_matrix,annot=True)


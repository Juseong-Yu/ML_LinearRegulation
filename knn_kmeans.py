import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

d_le = [75,77,83,81,73,99,72,83]
d_he = [24,29,19,32,21,22,19,34]

s_le = [76,78,82,88,76,83,81,89]
s_he = [55,58,53,54,61,52,57,64]

m_le= [35,39,38,41,30,57,41,35]
m_he = [23,26,19,30,21,24,28,20]

d_data = np.column_stack((d_le,d_he))
s_data = np.column_stack((s_le,s_he))
m_data = np.column_stack((m_le,m_he))

d_label = np.full(len(d_data),0)
s_label = np.full(len(s_data),1)
m_label = np.full(len(m_data),2)

dogs = np.concatenate((d_data,s_data,m_data))
labels = np.concatenate((d_label,s_label,m_label))
dog_class = {0:'닥스훈트',1:'사모예드',2:'말티즈'}

print("닥스훈트(0) : ",d_data.tolist())
print("사모예드(1) : ",s_data.tolist())
print("말티즈(2) : ",m_data.tolist())
print()

k = 3
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(dogs,labels)
y_pred_all = knn.predict(dogs)

conf_mat = confusion_matrix(labels,y_pred_all)
print(conf_mat)

A = [[58,30]]
B = [[80,26]]
C = [[80,41]]
D = [[75,55]]

y_pred_a1 = knn.predict(A)
y_pred_b1 = knn.predict(B)
y_pred_c1 = knn.predict(C)
y_pred_d1 = knn.predict(D)

k = 5
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(dogs,labels)
y_pred_a2 = knn.predict(A)
y_pred_b2 = knn.predict(B)
y_pred_c2 = knn.predict(C)
y_pred_d2 = knn.predict(D)

k = 7
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(dogs,labels)
y_pred_a3 = knn.predict(A)
y_pred_b3 = knn.predict(B)
y_pred_c3 = knn.predict(C)
y_pred_d3 = knn.predict(D)

print("A 데이터 분석 결과")
print("A",A,":n_neighbors가 3일때 :",dog_class[y_pred_a1[0]])
print("A",A,":n_neighbors가 5일때 :",dog_class[y_pred_a2[0]])
print("A",A,":n_neighbors가 7일때 :",dog_class[y_pred_a3[0]])
print()

print("B 데이터 분석 결과")
print("B",B,":n_neighbors가 3일때 :",dog_class[y_pred_b1[0]])
print("B",B,":n_neighbors가 5일때 :",dog_class[y_pred_b2[0]])
print("B",B,":n_neighbors가 7일때 :",dog_class[y_pred_b3[0]])
print()

print("C 데이터 분석 결과")
print("C",C,":n_neighbors가 3일때 :",dog_class[y_pred_c1[0]])
print("C",C,":n_neighbors가 5일때 :",dog_class[y_pred_c2[0]])
print("C",C,":n_neighbors가 7일때 :",dog_class[y_pred_c3[0]])
print()

print("D 데이터 분석 결과")
print("D",D,":n_neighbors가 3일때 :",dog_class[y_pred_d1[0]])
print("D",D,":n_neighbors가 5일때 :",dog_class[y_pred_d2[0]])
print("D",D,":n_neighbors가 7일때 :",dog_class[y_pred_d3[0]])
print()

plt.scatter(d_le,d_he,c='red',label='Dachshund')
plt.scatter(s_le,s_he,c='blue',marker='^',label='Samoyed')
plt.scatter(m_le,m_he,c='green',marker='s',label='Maltese')
plt.scatter(A[0][0],A[0][1],c = 'purple',s = 500)
plt.scatter(B[0][0],B[0][1],c = 'gray',s = 500)
plt.scatter(C[0][0],C[0][1],c = 'c',s = 500)
plt.scatter(D[0][0],D[0][1],c = 'green',s = 500)
plt.show()
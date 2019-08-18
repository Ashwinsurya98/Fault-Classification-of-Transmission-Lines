#'importing required packages'
import scipy.io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split 


#'loading the data from MATLAB'
data=scipy.io.loadmat ('Data Set workspace.mat')
inputs=data['inputs'].T
sinputs=data['scaledI'].T
target=data['Y'].T
out=np.random.randint(0, 1, size=(150012, 1))
out[0:10001]=1
out[10002:20003]=2
out[20003:30004]=3
out[30004:40005]=4
out[40005:50006]=5
out[50006:60007]=6
out[60007:70008]=7
out[70008:80009]=8
out[80009:90010]=9
out[90010:100011]=10
out[100011:]=11

#'Creating Dataframe'
df = pd.DataFrame(np.concatenate((sinputs, target,out), 1))
df.columns=['Va','Vb','Vc','Ia','Ib','Ic','ag','bg','cg','abg','acg','bcg','abc','ab','ac','bc','no','tar']
X=df[['Va','Vb','Vc','Ia','Ib','Ic']]
y=df[['tar']]
out=out.ravel()

#Splitting the Data
X_train, X_test, y_train, y_test = train_test_split(X, out, random_state = 0)

# training a linear SVM classifier 
from sklearn.svm import SVC 
svm_model_linear = SVC(kernel = 'linear', C = 1).fit(X_train, y_train) 
svm_predictions = svm_model_linear.predict(X_test) 

# model accuracy for X_test   
accuracy = svm_model_linear.score(X_test, y_test) 
  
# creating a confusion matrix 
cm = confusion_matrix(y_test, svm_predictions) 
a=cm.sum()
accuracy = (cm[0][0]+cm[1][1]+cm[2][2]+cm[3][3]+cm[4][4]+cm[5][5]+cm[6][6]+cm[7][7]+cm[8][8]+cm[9][9]+cm[10][10])/a
print("Accuracy: "+ str(accuracy*100)+"%")

#gridsearch cv
from sklearn.model_selection import GridSearchCV
param_grid={'C':[0.1,1,10,100,1000],'gamma':[1,0.1,0.01,0.001,0.0001]}
grid=GridSearchCV(SVC(),param_grid,verbose=3)
grid.fit(X_train,y_train)



from sklearn.svm import SVC
clf = SVC(gamma='auto')
clf.fit(X_train, y_train)
y_pred=clf.predict(X_test)
cm = confusion_matrix(y_test,y_pred) 
a=cm.sum()
accuracy = (cm[0][0]+cm[1][1]+cm[2][2]+cm[3][3]+cm[4][4]+cm[5][5]+cm[6][6]+cm[7][7]+cm[8][8]+cm[9][9]+cm[10][10])/a
print("Accuracy: "+ str(accuracy*100)+"%")



from sklearn.svm import NuSVC
clf = NuSVC(gamma='scale', nu=0.1)
clf.fit(X_train, y_train)
y_pred=clf.predict(X_test)
cm = confusion_matrix(y_test,y_pred) 
a=cm.sum()
accuracy = (cm[0][0]+cm[1][1]+cm[2][2]+cm[3][3]+cm[4][4]+cm[5][5]+cm[6][6]+cm[7][7]+cm[8][8]+cm[9][9]+cm[10][10])/a
print("Accuracy: "+ str(accuracy*100)+"%")


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

# training a Naive Bayes classifier 
from sklearn.naive_bayes import GaussianNB 
gnb = GaussianNB().fit(X_train, y_train) 
gnb_predictions = gnb.predict(X_test) 
  
# accuracy on X_test 
accuracy = gnb.score(X_test, y_test) 
print (accuracy) 
  
# creating a confusion matrix 
cm = confusion_matrix(y_test, gnb_predictions) 
a=cm.sum()
accuracy = (cm[0][0]+cm[1][1]+cm[2][2]+cm[3][3]+cm[4][4]+cm[5][5]+cm[6][6]+cm[7][7]+cm[8][8]+cm[9][9]+cm[10][10])/a
print("Accuracy: "+ str(accuracy*100)+"%")


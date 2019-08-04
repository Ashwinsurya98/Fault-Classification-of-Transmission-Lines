#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as pld
import scipy.io


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
y=df[['ag','bg','cg','abg','acg','bcg','abc','ab','ac','bc','no']]


#Splitting into training and test sets
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.25, random_state=0)

#import required libraries
import keras
from keras.models import Sequential
from keras.layers import Dense 
from keras.layers import Dropout

#tuning parameters
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense

def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = 12, kernel_initializer = 'uniform', activation = 'relu', input_dim = 6))
    classifier.add(Dense(units = 9, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 11, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'mean_squared_error', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size': [1, 5],
              'epochs': [10, 20],
              'optimizer': ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_
print("best_parameters: ")
print(best_parameters)
print("\nbest_accuracy: ")
print(best_accuracy)

#building ANN
from keras.models import Sequential
from keras.layers import Dense
# Initialising the ANN
classifier = Sequential() 
#Adding the input layer and the first hidden layer
classifier.add(Dense(units = 12, kernel_initializer = 'uniform', activation = 'relu', input_dim = 6))
#Adding second hidden layer
classifier.add(Dense(units = 9, kernel_initializer = 'uniform', activation = 'relu'))
#Adding output layer
classifier.add(Dense(units = 11, kernel_initializer = 'uniform', activation = 'sigmoid'))

#compiling ANN
classifier.compile(optimizer = 'rmsprop', loss = 'mean_squared_error', metrics = ['accuracy'])

#fitting ANN
classifier.fit(X_train, y_train, batch_size = 5, epochs = 20)

#predictions
y_pred = classifier.predict(X_test)

#confusion matrix and accuracy
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test.values.argmax(axis=1), y_pred.argmax(axis=1))
print(cm)
a=cm.sum()
accuracy = (cm[0][0]+cm[1][1]+cm[2][2]+cm[3][3]+cm[4][4]+cm[5][5]+cm[6][6]+cm[7][7]+cm[8][8]+cm[9][9]+cm[10][10])/a
print("Accuracy: "+ str(accuracy*100)+"%")

#saving the model
import pickle
filename1='ann_model.sav'
pickle.dump(classifier,open(filename1,'wb'))

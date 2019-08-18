import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
out1=np.random.randint(0, 1, size=(150012, 1))
out1[0:10001]=1
out1[60007:70008]=1
out2=np.random.randint(0, 1, size=(20002, 1))
out2[0:10001]=1
out3=np.random.randint(0, 1, size=(20002, 1))
out3[10001:]=1
out4=np.random.randint(0, 1, size=(20002, 1))
out4[0:10001]=0
out4[10001:]=1
out4=out4.ravel()
#'Creating Dataframe'
df = pd.DataFrame(np.concatenate((sinputs, target,out), 1))
df.columns=['Va','Vb','Vc','Ia','Ib','Ic','ag','bg','cg','abg','acg','bcg','abc','ab','ac','bc','no','tar']
X=df[['Va','Vb','Vc','Ia','Ib','Ic']]
y=df.drop(columns=['ag','abc','Va','Vb','Vc','Ia','Ib','Ic','tar'])
output1=pd.DataFrame(np.concatenate((y,out1), 1))
output1.columns=['bg','cg','abg','acg','bcg','ab','ac','bc','no','ag-abc']
output2=pd.DataFrame(np.concatenate((out2,out3), 1))
output2.columns=['ag','abc']
inp=pd.DataFrame(np.concatenate((X[0:10001],X[60007:70008]), 0))
inp.columns=['Va','Vb','Vc','Ia','Ib','Ic']
#Splitting into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X, output1, test_size=0.25, random_state=0)
classif=onelevelclassifier(X_train, X_test, y_train, y_test)
pred=classif.predict(inpset)
fp=pred.argmax(axis=1)
for i in range(1,10):
    if fp[i]==9:
        clf=twolevelclassifier(inp,output2)
        pred2=clf.predict(pd.DataFrame(inpset.iloc[i]).transpose())
        plt.plot(pred2.argmax(axis=1),1,marker='o')
plt.plot(fp,ans,marker='x')
inpset.iloc[9]
ans=[]
for i in range(0,10):
    for j in range(0,10):
        if(testset[j].iloc[i]==1):
            ans.append(j)
o=pd.DataFrame(inpset.iloc[9])
o.transpose()           
testset=pd.DataFrame()
testset[0]=y_test.loc[17860]
testset[1]=y_test.loc[25821]
testset[2]=y_test.loc[37661]
testset[3]=y_test.loc[45090]
testset[4]=y_test.loc[58524]
testset[5]=y_test.loc[72592]
testset[6]=y_test.loc[84647]
testset[7]=y_test.loc[92012]
testset[8]=y_test.loc[129119]
testset[9]=y_test.loc[64107]
w=[[0.19469238560690608,0.4849507531122468,0.6601116246499921,0.48498005770661223,0.49948687062282254,0.48497959228737314],
   [0.18191497168878545,0.605293225724762,0.4850063440272118,0.48498007051182157,0.4849796471335675,0.4716913793012772],
   [0.48497355483842985,0.48495502241628813,0.31874357903752176,0.480873103073063,0.49940552523278053,0.48497993365960673],
   [0.4850046031635826,0.3540871407440065,0.4849889877155671,0.47149599783555035,0.48497989833771626,0.48711144586718763],
   [0.19787874259884244,0.48496882740629765,0.4849536675390752,0.4849800545224157,0.48357305847100784,0.4987329257086071],
   [0.3971659986189834,0.3972029338044414,0.6605706408880369,0.503447535945356,0.46651235010359476,0.48497950404756307],
   [0.4263502043288199,0.6022200281137892,0.4263693824736593,0.49456895733492845,0.484979591573503,0.47539081201275524],
   [0.17628593252699856,0.6393230076407704,0.6393299791059867,0.4849802304742981,0.4884651913341263,0.4814937261460628],
   [0.7148480738683847,0.390926660180736,0.3491645682540009,0.4849795375654311,0.4849798614868448,0.48497990324893675],
   [0.4850081626942452,0.6064356505803976,0.6548906688147161,0.47078228284178286,0.48497964599205223,0.48497959753703396]
   ]
inpset=pd.DataFrame(w)
inpset.columns=['Va','Vb','Vc','Ia','Ib','Ic']
print("Accuracy: "+ str(accuracy*100)+"%")
s= X_test.iloc[5]
def onelevelclassifier(X_train, X_test, y_train, y_test):
    
    

    #random forest classifier intialization
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=100,random_state=0)
 
    #fitting model
    clf.fit(X_train, y_train)
    return clf

    #Predicting
    y_pred=clf.predict(X_test)


    #confusion matrix and accuracy
    from sklearn.metrics import confusion_matrix
    cm=confusion_matrix(y_test.values.argmax(axis=1),y_pred.argmax(axis=1))

    a=cm.sum()
    accuracy = (cm[0][0]+cm[1][1]+cm[2][2]+cm[3][3]+cm[4][4]+cm[5][5]+cm[6][6]+cm[7][7]+cm[8][8]+cm[9][9])/a
   
    return y_pred
accuracy2=twolevelclassifier(inp,output2)
print("Accuracy: "+ str(accuracy2*100)+"%")
########################################################################################
def twolevelclassifier(inp,output2):
    #Splitting into training and test sets
    from sklearn.model_selection import train_test_split
    X2_train, X2_test, y2_train, y2_test=train_test_split(inp, output2, test_size=0.25, random_state=0)

    #random forest classifier intialization
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=200,random_state=0)
 
    #fitting model
    clf.fit(X2_train, y2_train)
    return clf
filename1='1class_model.sav'
pickle.dump(clf,open(filename1,'wb'))
filename='2class_model.sav'
pickle.dump(gnb,open(filename,'wb'))
import pickle
classif=pickle.load(open('2class_model.sav','rb'))


    #Predicting
    y_pred=clf.predict(X2_test)

    #confusion matrix and accuracy
    from sklearn.metrics import confusion_matrix
    cm=confusion_matrix(y2_test.values.argmax(axis=1), y_pred.argmax(axis=1))
    
    a=cm.sum()
    accuracy = (cm[0][0]+cm[1][1])/a
    return accuracy
#############################################################################
def twolevelclassifier(inp,output2):
    
    #Splitting into training and test sets
    from sklearn.model_selection import train_test_split
    X2_train, X2_test, y2_train, y2_test=train_test_split(inp, out4, test_size=0.25, random_state=0)
    # training a Naive Bayes classifier 
    from sklearn.naive_bayes import GaussianNB 
    gnb = GaussianNB().fit(X2_train, y2_train) 
    return gnb
    gnb_predictions = gnb.predict(X2_test) 
        
    # creating a confusion matrix 
    from sklearn.metrics import confusion_matrix
    cm2 = confusion_matrix(y2_test, gnb_predictions) 
    a2=cm2.sum()
    accuracy2 = (cm2[0][0]+cm2[1][1])/a2
    print("Accuracy: "+ str(accuracy2*100)+"%")

###########################################################################
###SVM
#gridsearch cv
from sklearn.model_selection import GridSearchCV
param_grid={'C':[0.1,1,10,100,1000],'gamma':[1,0.1,0.01,0.001,0.0001]}
grid=GridSearchCV(SVC(),param_grid,verbose=3)
grid.fit(X2_train,y2_train)
bestaccuracy=grid.best_score_
param=grid.best_params_
from sklearn.svm import SVC
model=SVC(kernel='linear',C=0.1,gamma=0.1)
model.fit(X2_test,y2_test)
 #Predicting
y_pred=model.predict(X2_test)

    #confusion matrix and accuracy
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y2_test, y_pred)
    
a=cm.sum()
accuracy = (cm[0][0]+cm[1][1])/a
cm3=cm


##############################################################################
# training a KNN classifier 
from sklearn.neighbors import KNeighborsClassifier 
knn = KNeighborsClassifier(n_neighbors = 10).fit(X2_train, y2_train)
# creating a confusion matrix 
knn_predictions = knn.predict(X2_test)  
cm = confusion_matrix(y2_test, knn_predictions) 
a=cm.sum()
accuracy = (cm[0][0]+cm[1][1])/a


w=[[0.48524407745096726,0.35036951940306676,0.31449897708469865,0.3528246060727903,0.48497990217614045,0.4849799380466828],
   [0.4850081626942452,0.6064356505803976,0.6548906688147161,0.47078228284178286,0.48497964599205223,0.48497959753703396]]
inpset=pd.DataFrame(w)
inpset.columns=['Va','Vb','Vc','Ia','Ib','Ic']
ans2=[0,1]
pred=gnb.predict(inpset)

#plotting the plots
plt.plot(pred,ans,marker='o',markersize=7)          
plt.xlabel("Actual outputs")
plt.ylabel("Predicted outputs")
plt.title("Plot Between Actual V/S Predicted Values")  




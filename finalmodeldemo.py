import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io

#preparing inputs from testset
w=[[ 0.484964 ,   0.617941  ,  0.653916  ,  0.492664 ,   0.484980 ,   0.484980],
   [0.19469238560690608,0.4849507531122468,0.6601116246499921,0.48498005770661223,0.49948687062282254,0.48497959228737314],
   [0.18191497168878545,0.605293225724762,0.4850063440272118,0.48498007051182157,0.4849796471335675,0.4716913793012772],
   [0.48497355483842985,0.48495502241628813,0.31874357903752176,0.480873103073063,0.49940552523278053,0.48497993365960673],
   [0.4850046031635826,0.3540871407440065,0.4849889877155671,0.47149599783555035,0.48497989833771626,0.48711144586718763],
   [0.19787874259884244,0.48496882740629765,0.4849536675390752,0.4849800545224157,0.48357305847100784,0.4987329257086071],
   [0.4850081626942452,0.6064356505803976,0.6548906688147161,0.47078228284178286,0.48497964599205223,0.48497959753703396],
   [0.3971659986189834,0.3972029338044414,0.6605706408880369,0.503447535945356,0.46651235010359476,0.48497950404756307],
   [0.4263502043288199,0.6022200281137892,0.4263693824736593,0.49456895733492845,0.484979591573503,0.47539081201275524],
   [0.17628593252699856,0.6393230076407704,0.6393299791059867,0.4849802304742981,0.4884651913341263,0.4814937261460628],
   [0.7148480738683847,0.390926660180736,0.3491645682540009,0.4849795375654311,0.4849798614868448,0.48497990324893675],
   ]
inpset=pd.DataFrame(w)
inpset.columns=['Va','Vb','Vc','Ia','Ib','Ic']

#Actual outputs 
testset=pd.DataFrame()
testset[0]=y_test.loc[6553]
testset[1]=y_test.loc[17860]
testset[2]=y_test.loc[25821]
testset[3]=y_test.loc[37661]
testset[4]=y_test.loc[45090]
testset[5]=y_test.loc[58524]
testset[6]=y_test.loc[64107]
testset[7]=y_test.loc[72592]
testset[8]=y_test.loc[84647]
testset[9]=y_test.loc[92012]
testset[10]=y_test.loc[129119]
X_test.iloc[125]
ans=[]
for i in range(0,11):
    for j in range(0,11):
        if(testset[j].iloc[i]==1):
            ans.append(j)

#importing the classifier
            #filename='rf_model.sav'
             #pickle.dump(clf,open(filename,'wb'))
import pickle
classif=pickle.load(open('rf_model.sav','rb'))

#predicting the values
pred=classif.predict(inpset)

#plotting the plots
plt.plot(pred.argmax(axis=1),ans,marker='x',markersize=7,label='0-ag\n1-bg\n2-cg\n3-abg\n4-acg\n5-bcg\n6-abc\n7-ab\n8-ac\n9-bc\n10-no')          
plt.xlabel("Actual outputs")
plt.ylabel("Predicted outputs")
plt.title("Plot Between Actual V/S Predicted Values")  
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))


import numpy as np
import pandas as pd
import pickle
df=pd.read_csv("creditcard.csv")

from sklearn.preprocessing import StandardScaler
df['NormalizedAmount'] = StandardScaler().fit_transform(df['Amount'].values.reshape(-1,1))
df = df.drop(['Time','Amount'], axis=1)
print("1")
X = np.array(df.loc[:,df.columns != 'Class'])
y = np.array(df.loc[:,df.columns == 'Class'])
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.2,random_state=0)
print("2")
sm = SMOTE(random_state=2)
X_train_res, y_train_res = sm.fit_sample(X_train, y_train.ravel())

rfc=RandomForestClassifier(n_estimators=30,max_depth=50,n_jobs=-1)
rfc.fit(X_train_res,y_train_res.ravel())

pickle.dump(rfc, open('model.pkl','wb'))
model =pickle.load(open('model.pkl','rb'))
"""(model.predict[[-15.819178720771802,8.7759971528627,-22.8046864614815,11.864868080360699,-9.09236053189517,
                     -2.38689320657655,-16.5603681078199,0.9483485947860579,-6.31065843275059,-13.0888909176936,
                     9.81570317447819,-14.0560611837648,0.777191846436601,-13.7610179615936,-0.353635939812489,
                     -7.9574472262599505,-11.9629542349435,-4.7805077876172,0.652498045264831,0.992278949261366,
                     -2.35063374523783,1.03636187430048,1.13605073696052,-1.0434137405139001,-0.10892334328197999,
                     0.657436778462222,2.1364244708551396,-1.41194537483904,-0.3492313067728856]])"""
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 14:42:54 2021

@author: HEMA
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler  
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import pickle
df=pd.read_csv("C:/Users\Gayathri\Documents\liver disease\indian_liver_patient.csv")
categorical=['Gender']
numerical=['Age','Total_Bilirubin', 'Direct_Bilirubin',
       'Alkaline_Phosphotase', 'Alamine_Aminotransferase',
       'Aspartate_Aminotransferase', 'Total_Protiens', 'Albumin',
       'Albumin_and_Globulin_Ratio','Dataset']
cols=['Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin',
       'Alkaline_Phosphotase', 'Alamine_Aminotransferase',
       'Aspartate_Aminotransferase', 'Total_Protiens', 'Albumin',
       'Albumin_and_Globulin_Ratio', 'Dataset']
for i in cols:
    df[i]=df[i].fillna(df[i].dropna().mode()[0])
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
for i in categorical:
    df[i] = le.fit_transform(df[i])

    
train=df.iloc[:,0:1]
test=df.iloc[:,-1]



from sklearn import preprocessing
y=df.Dataset
x=df.drop('Dataset',axis=1)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
st_x= preprocessing.StandardScaler()
x_train= st_x.fit_transform(x_train)  
x_test= st_x.transform(x_test)

lg=RandomForestClassifier()
k=lg.fit(x_train,y_train)

pickle.dump(lg, open('.\model\model.pkl','wb'))
model = pickle.load(open('.\model\model.pkl','rb'))


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler  
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import pickle
df=pd.read_csv(".\dataset\indian_liver_patient.csv")

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
'''x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
st_x= preprocessing.StandardScaler()
x_train= st_x.fit_transform(x_train)  
x_test= st_x.transform(x_test)

from sklearn.model_selection import train_test_split'''
  
# describes info about train and test set
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
print("Number transactions X_train dataset: ", x_train.shape)
print("Number transactions y_train dataset: ", y_train.shape)
print("Number transactions X_test dataset: ", x_test.shape)
print("Number transactions y_test dataset: ", y_test.shape)

lg=RandomForestClassifier()
k=lg.fit(x_train,y_train)


print("Before OverSampling, counts of label '1': {}".format(sum(y_train == 1)))
print("Before OverSampling, counts of label '02: {} \n".format(sum(y_train == 2)))
  
# import SMOTE module from imblearn library
# pip install imblearn (if you don't have imblearn in your system)
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state = 2)
x_train_res, y_train_res = sm.fit_resample(x_train, y_train.ravel())
  
print('After OverSampling, the shape of train_X: {}'.format(x_train_res.shape))
print('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape))
  
print("After OverSampling, counts of label '1': {}".format(sum(y_train_res == 1)))
print("After OverSampling, counts of label '2': {}".format(sum(y_train_res == 2)))

pickle.dump(lg, open('.\model\model.pkl','wb'))
model = pickle.load(open('.\model\model.pkl','rb'))

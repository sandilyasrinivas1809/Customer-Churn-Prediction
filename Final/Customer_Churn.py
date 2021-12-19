
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
import pickle
from sklearn.preprocessing import LabelEncoder


df = pd.read_csv(r'C:\Users\User\Documents\College\Projects\Major project\Customer Churn\Churn_Modelling.csv')

df1 = df[['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Exited']]

le=LabelEncoder()
for i,v in enumerate(['Geography', 'Gender']):

    df1.loc[:,v]=le.fit_transform(df1.loc[:,v])
    df_LE = df1.copy()

X = df_LE.iloc[:, [0,1,2,3,4,5,6,7,8,9]]
y = df_LE.iloc[:, 10].values

smpl = SMOTE()
X_res,y_res=smpl.fit_resample(X,y)

X_train, X_test, y_train, y_test = train_test_split(X_res,y_res,test_size=0.2, random_state = 42)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


gb = GradientBoostingClassifier()
gb.fit(X_train,y_train)

y_pred = gb.predict(X_test)

pickle_out = open("Customer_Churn_pred_gb.pkl", "wb") 
pickle.dump(gb, pickle_out) 
pickle_out.close()

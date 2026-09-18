import pandas as pd
import joblib

#pip install scikit-learn
#pip install pandas
#pip install joblib

data=pd.read_csv('Salary_Data.csv')
avgsal=data['Salary'].mean()

data.fillna({'Salary':avgsal},inplace=True)

x=data.iloc[:,:-1].values
y=data.iloc[:,-1].values

from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x,y)

joblib.dump(model,"ml65.joblib")
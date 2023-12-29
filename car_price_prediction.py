import pandas as pd
car=pd.read_csv("C:\\Users\\Ratna Shekar Reddy\\Documents\\Projects\\ML\\cleaned car dataset.csv")


print("*********INFORMATION OF DATASET***********\n")
car.info()

print("*************")
car['Price'].unique()
"""##Quality
.year has many non year values
.year object to int
.price has ask for price
.price object to int
.kms_driven has kms with integers
.kms driven object to int
.kms has no values
.fuel_type has nan values
.keepfirst words of name"""

#cleaning
Backup=car.copy()
car=car[car['year'].apply(str).str.isnumeric()]
car['year']=car['year'].astype(int)
car.info()

car=car[car['Price']!="Ask For Price"]

car['Price']=car['Price'].replace(',','').astype(int)
car.info()
car['kms_driven']=car['kms_driven'].apply(str).str.split(' ').apply(str).apply(str).get(0).replace(',','')
car=car[car['kms_driven'].str.isnumeric()]
car['kms_driven']=car['kms_driven'].astype(int)
car.info()

car=car[car['fuel_type']!='nan']
car.info()
car['name']=car['name'].str.split(' ').str.slice(0,3).str.join(' ')
car.reset_index(drop=True)
car.describe()
car=car[car['Price']<6e6].reset_index(drop=True)
car.to_csv('cleaned car dataset.csv')
#Model
x=car.drop(columns='Price')
y=car['Price']

!pip install --upgrade scikit-learn


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
#OneHotEncoder is used to convert the data in to machine understanding (computer code)
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
ohe=OneHotEncoder()
ohe.fit(x[['name','company','fuel_type']])
ohe.categories_
column_trans=make_column_transformer((OneHotEncoder(categories=ohe.categories_),['name','company','fuel_type']),remainder="passthrough")
lr=LinearRegression()
pipe=make_pipeline(column_trans,lr)
pipe.fit(x_train,y_train)

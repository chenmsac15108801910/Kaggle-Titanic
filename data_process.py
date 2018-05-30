import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def processdata(csvdata):
    train_data = pd.read_csv('./Data/'+csvdata+'.csv')

    #Find title in "Name"
    select_data = pd.DataFrame(train_data)
    select_data.loc[select_data['Name'].str.contains('Mr.'),'Mr'] = 1
    select_data.loc[select_data['Name'].str.contains('Miss.'),'Miss'] = 1
    select_data.loc[select_data['Name'].str.contains('Mrs.'),'Mrs'] = 1
    select_data.loc[select_data['Name'].str.contains('Master.'),'Master'] = 1

    #Split Pclass to Pclass_1 to 3
    select_data.loc[select_data['Pclass'] == 1, 'Pclass_1'] = 1
    select_data.loc[select_data['Pclass'] == 2, 'Pclass_2'] = 1
    select_data.loc[select_data['Pclass'] == 3, 'Pclass_3'] = 1

    #Change Embarked to Embarked_S, C, Q
    select_data['Embarked'] = select_data["Embarked"].fillna('S')
    select_data.loc[select_data['Embarked'] == "S", 'Embarked_S'] = 1
    select_data.loc[select_data['Embarked'] == "C", 'Embarked_C'] = 1
    select_data.loc[select_data['Embarked'] == "Q", 'Embarked_Q'] = 1

    #Drop non-useful data
    drop_title = ['Embarked','Pclass','Name','Cabin','Ticket','Parch','SibSp']
    select_data = select_data.drop(drop_title, axis=1)

    #Change Sex to number
    select_data.loc[select_data['Sex'] == "male",'Sex'] = 0
    select_data.loc[select_data['Sex'] == "female",'Sex'] = 1
    select_data['Age'] = select_data["Age"].fillna(select_data["Age"].median())
    select_data['Fare'] = select_data["Fare"].fillna(select_data["Fare"].median())

    #Scalar Age and Fare data
    scalar=MinMaxScaler()
    select_data["Age"]=scalar.fit_transform(select_data["Age"].values.reshape(-1,1))
    select_data["Fare"] = scalar.fit_transform(select_data["Fare"].values.reshape(-1, 1))

    #fill all NaN data to 0
    select_data = select_data.fillna(0)

    print(select_data)
    select_data.to_csv('Processed'+csvdata+'.csv',index = False)

    
processdata('train')
processdata('test')

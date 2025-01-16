# ! pip install pandas

import pandas as pd
import numpy as np 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pickle 

def create_model(data):
    X = data.drop('diagnosis',axis=1)
    y = data['diagnosis']

    #scale the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    #split the data
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

    model = LogisticRegression()
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)

    print("Accuracy score of our Model: ",accuracy_score(y_test,y_pred))
    print("Classification Report: ",classification_report(y_test,y_pred))

    return model, scaler


def get_clean_data():
    data = pd.read_csv('data/data.csv')
    
    data = data.drop(['Unnamed: 32','id'],axis=1)
    data['diagnosis'] = data['diagnosis'].map({'M':1,'B':0})

    return data 

def main():
    data = get_clean_data()
    model, scaler = create_model(data)
    # test_model(model)


    with open('model/model.pkl', 'wb') as f :
        pickle.dump(model,f)

    with open('model/scaler.pkl','wb') as f:
        pickle.dump(scaler,f)    
    


if __name__ == '__main__':
    main()






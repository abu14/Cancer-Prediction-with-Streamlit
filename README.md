# Hi, I'm Abenezer Tesfaye! 👋

## 🚀 About Me
I'm a Machine Learning Engineer / Data Analyst

Currently working as a Data Analytics Developer.

# Overview

Welcome to my breast cancer prediction project! This project utilizes a logistic regression machine learning model to predict whether a breast mass/ tumour is **benign** or **malignant** based on measurements from a cytology lab.

With the interactive interface powered by Streamlit, I was able to demonstrate how different measurements impact the predictions. Using the sliders in the sidebar to manually update the inputs, and the app will dynamically recalculate the results based on the historical data of which it has been trained on.

The model delivers accurate predictions, offering valuable insights to healthcare professionals for making informed decisions.

Feel free to check out the app at the link here [Cancer Prediction App - Abenezer Tesfaye](https://cancer-prediction-model-abenezer-v1.streamlit.app) and experiment with different values to observe how changes influence the predicted outcomes!

## Demo

![dashboard](https://github.com/user-attachments/assets/ff124e6e-f0f5-477e-8e22-21e4a53dc471)


## Tools Used

- **Python:** allows me to analyze the data and find insights and patterns. Made use of the following Python libraries:
    - **Pandas Library** 
    - **Sklearn** 
    - **Numpy**  
- **Visual Studio Code:** Executing my Python scripts.
- **Git & GitHub:** For version control and sharing my Python code and analysis.



## Data

### Import Libraries
```python
import pandas as pd
import numpy as np 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pickle 

```


### Import & Clean Up Data

> Data Preparation and Cleanup

This section outlines the steps taken to prepare the data for analysis, ensuring accuracy and usability.

```python
    def get_clean_data():
        data = pd.read_csv('data/data.csv')
        
        data = data.drop(['Unnamed: 32','id'],axis=1)
        data['diagnosis'] = data['diagnosis'].map({'M':1,'B':0})

        return data 

    def main():
        data = get_clean_data()
        model, scaler = create_model(data)
        # test_model(model)
```


### Create Model and Scaler

```python
def create_model(data):
    X = data.drop('diagnosis',axis=1)
    y = data['diagnosis']

    #scale the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    #split the data
    X_train,X_test,y_train,y_test = train_test_split(
        X,y,test_size=0.2,         
        random_state=42)

    model = LogisticRegression()
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)

    print("Accuracy score of our Model: ",accuracy_score(y_test,y_pred))
    print("Classification Report: ",classification_report(y_test,y_pred))

    return model, scaler
```


### Save Model

```python
    with open('model/model.pkl', 'wb') as f :
        pickle.dump(model,f)

    with open('model/scaler.pkl','wb') as f:
        pickle.dump(scaler,f)    
```




## App

### Create Page

```python
st.set_page_config(
            page_title="Breast Cancer Predictor App",
            page_icon="👩‍⚕️",
            layout="wide",
            initial_sidebar_state="expanded"
        )

def add_sidebar():
    st.sidebar.header("Cell Nuclei Measurements")
    #st.sidebar.subheader("Breast Cancer Parameters Input Features")
    data = get_clean_data()        
```


### Slider Section

```python
def add_sidebar():
    st.sidebar.header("Cell Nuclei Measurements")
   
    data = get_clean_data() 

    input_dict = {}

    data = get_clean_data()


    for label, key in slider_labels:
        input_dict[key]= st.sidebar.slider(
            label, 
            min_value = float(0), 
            max_value= float(data[key].max()),
            value = float(data[key].mean())
            )
    return  input_dict

```


### Make Predictions
```python
def add_predictions(input_data):
    model = pickle.load(open('model/model.pkl', 'rb'))
    scaler = pickle.load(open('model/scaler.pkl', 'rb'))

    # Convert input_data to DataFrame with correct feature names
    input_df = pd.DataFrame([input_data])

    # Scale the input data
    input_array_scaled = scaler.transform(input_df)

    # Make predictions
    prediction = model.predict(input_array_scaled)

    st.header("Cell Cluster Prediction")
    st.write("Cell Cluster is: ")
    if prediction[0] == 0:
        st.write("Benign")
    else:
        st.write("Malignant")

    proba = model.predict_proba(input_array_scaled)
    st.write("Probability of being Benign: ", proba[0][0])
    st.write("Probability of being Malignant: ", proba[0][1])

```

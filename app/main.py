import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go


st.set_page_config(
            page_title="Breast Cancer Predictor App",
            page_icon="👩‍⚕️",
            layout="wide",
            initial_sidebar_state="expanded"
        )


def get_clean_data():
    data = pd.read_csv('data/data.csv')
    data = data.drop(columns=['Unnamed: 32','id'],axis=1)
    data['diagnosis'] = data['diagnosis'].map({'M':1,'B':0})
    return data 


def add_sidebar():
    st.sidebar.header("Cell Nuclei Measurements")
    #st.sidebar.subheader("Breast Cancer Parameters Input Features")
    data = get_clean_data()

    slider_labels = [('Radius (Mean)', 'radius_mean'),
                    ('Texture (Mean)', 'texture_mean'),
                    ('Perimeter (Mean)', 'perimeter_mean'),
                    ('Area (Mean)', 'area_mean'),
                    ('Smoothness (Mean)', 'smoothness_mean'),
                    ('Compactness (Mean)', 'compactness_mean'),
                    ('Concavity (Mean)', 'concavity_mean'),
                    ('Concave Points (Mean)', 'concave points_mean'),
                    ('Symmetry (Mean)', 'symmetry_mean'),
                    ('Fractal Dimension (Mean)', 'fractal_dimension_mean'),
                    ('Radius (Se)', 'radius_se'),
                    ('Texture (Se)', 'texture_se'),
                    ('Perimeter (Se)', 'perimeter_se'),
                    ('Area (Se)', 'area_se'),
                    ('Smoothness (Se)', 'smoothness_se'),
                    ('Compactness (Se)', 'compactness_se'),
                    ('Concavity (Se)', 'concavity_se'),
                    ('Concave Points (Se)', 'concave points_se'),
                    ('Symmetry (Se)', 'symmetry_se'),
                    ('Fractal Dimension (Se)', 'fractal_dimension_se'),
                    ('Radius (Worst)', 'radius_worst'),
                    ('Texture (Worst)', 'texture_worst'),
                    ('Perimeter (Worst)', 'perimeter_worst'),
                    ('Area (Worst)', 'area_worst'),
                    ('Smoothness (Worst)', 'smoothness_worst'),
                    ('Compactness (Worst)', 'compactness_worst'),
                    ('Concavity (Worst)', 'concavity_worst'),
                    ('Concave Points (Worst)', 'concave points_worst'),
                    ('Symmetry (Worst)', 'symmetry_worst'),
                    ('Fractal Dimension (Worst)', 'fractal_dimension_worst')]    

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


def get_scaled_values(input_dict):
    data = get_clean_data()
    X = data.drop('diagnosis',axis=1)
    scaled_dict = {}
    for key, value in input_dict.items():
        min_value = X[key].min()
        max_value = X[key].max()
        scaled_value = (value - min_value) / (max_value - min_value)
        scaled_dict[key] = scaled_value

    return scaled_dict 


def get_radar_chart(input_data):
    
    input_data = get_scaled_values(input_data)
    
    categories = ['Radius', 'Texture', 'Perimeter', 'Area', 
                'Smoothness', 'Compactness', 
                'Concavity', 'Concave Points',
                'Symmetry', 'Fractal Dimension']
    

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_mean'], input_data['texture_mean'], 
            input_data['perimeter_mean'], input_data['area_mean'],
            input_data['smoothness_mean'], input_data['compactness_mean'],
            input_data['concavity_mean'], input_data['concave points_mean'],
            input_data['symmetry_mean'], input_data['fractal_dimension_mean']
           ],
        theta=categories,
        fill='toself',
        name='Mean Data'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_se'], input_data['texture_se'],
            input_data['perimeter_se'], input_data['area_se'],
            input_data['smoothness_se'], input_data['compactness_se'],
            input_data['concavity_se'], input_data['concave points_se'],
            input_data['symmetry_se'], input_data['fractal_dimension_se']
        ],
        theta=categories,
        fill='toself',
        name='Standard Error'
    ))

    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_worst'], input_data['texture_worst'],
            input_data['perimeter_worst'], input_data['area_worst'],
            input_data['smoothness_worst'], input_data['compactness_worst'],
            input_data['concavity_worst'], input_data['concave points_worst'],
            input_data['symmetry_worst'], input_data['fractal_dimension_worst']
        ],
        theta=categories,
        fill='toself',
        name='Worst Value'
    ))


    fig.update_layout(
    polar=dict(
        radialaxis=dict(
        visible=True,
        range=[0, 1]
        )),
    showlegend=True
    )
    return fig


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

    st.write("Please note that this app should be used to assist medical professionals in making a diagnosis, and not be used as a substitute for a professional diagnosis.")




def main():
    input_data = add_sidebar()
    
    with st.container():
        st.title("Breast Cancer Predictor App")
        st.write("Please connect this app to your cytology lab to help diagnose breast cancer form your tissue sample. This app predicts using a machine learning model whether a breast mass is benign or malignant based on the measurements it receives from your cytosis lab. You can also update the measurements by hand using the sliders in the sidebar. The model combines these measurements with historical data, the model provides accurate predictions that can assist healthcare professionals in making informed decisions.")
        #st.write("This application utilizes advanced algorithms to analyze various features of the breast mass, such as size, shape, and texture. ")
        st.write("We encourage you to input different values using the sliders to see how changes in the measurements affect the predicted outcome.")
    col1, col2 = st.columns((4,1))

    with col1:
        radar_chart = get_radar_chart(input_data)
        st.plotly_chart(radar_chart)
    with col2:
        add_predictions(input_data)    



if __name__ == '__main__':
    main()
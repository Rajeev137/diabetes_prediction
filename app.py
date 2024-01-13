import numpy as np
import pickle
import streamlit as st

#loading the saved model
loaded_model = pickle.load(open('/Users/rajeevsharma/Downloads/project/Diabetes prediction/trained_model.sav', 'rb'))

#creating a fuction fro prediction 
def diabetes_prediction(input_data):

    #changing the input data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    #reshape the array as we are predicting only for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction = loaded_model.predict(input_data_reshaped)

    if (prediction[0] == 0):
        return 'The person is not diabetic'
    else:
        return 'The person is diabetic'
    
def main():
    #giving the title
    st.title('Diabetes Prediction Web App')

    #getting the input data from the use
    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Glucose level')
    BloodPressure = st.text_input('Blood Pressure Value ')
    SkinThickness = st.text_input('Skin Thickneess Value')
    Insulin = st.text_input('Insulin Value')
    Bmi = st.text_input('BMI Value')
    DiabetesPedigreeFunction = st.text_input('Daibetes Pedigree Function Value')
    Age = st.text_input('Age of the Person')

    #creating a result string
    diagnosis  = ''
    if st.button("Diabetes Test Results"):
        diagnosis = diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, Bmi, DiabetesPedigreeFunction, Age])

    st.success(diagnosis)


if __name__ == '__main__':
    main()
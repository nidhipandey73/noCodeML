import os

import streamlit as st
#from streamlit_option_menu import option_menu
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


from ml_utility import(read_data, 
                       preprocess_data, 
                       train_model, 
                       evaluate_model)

# get the working directory of the main.py file 

working_dir= os.path.dirname(os.path.abspath(__file__))

# get the parent directory 

parent_dir= os.path.dirname(working_dir)

st.set_page_config(
    page_title = "Automate ML",
    layout = "centered"
)

st.title("Automate Machine Learning")

dataset_list = os.listdir(f"{parent_dir}/data")

dataset = st.selectbox("Select a dataset from the dropdown",
                       dataset_list,
                       index=None)

df = read_data(dataset)

if df is not None:
    st.dataframe(df.head())

    col1, col2, col3, col4 = st.columns(4)

    scaler_type_list = ["Standard", "MinMax"]

    model_dictionary = {
        "Logistic Regression" : LogisticRegression(),
        "Support Vector Machine" : SVC(),
        "Random Forest" : RandomForestClassifier(),
        "XGBoost" : XGBClassifier() 
    }

    with col1:
        target_column = st.selectbox("Select the target column", list(df.columns))
    with col2:
        scaler_type = st.selectbox("Select the scaler type", scaler_type_list)
    with col3:
        selected_model = st.selectbox("Select the model", list(model_dictionary.keys()))
    with col4:
        model_name = st.text_input("Model name")

    if st.button("Train the Model"):
        X_train, X_test, Y_train, Y_test = preprocess_data(df, target_column, scaler_type)
        model_to_be_trained = model_dictionary[selected_model]
        model = train_model(X_train, Y_train, model_to_be_trained, model_name)
        accuracy = evaluate_model(model, X_test, Y_test)
        
        st.success("Accuracy on Test Data:" + str(accuracy))




    
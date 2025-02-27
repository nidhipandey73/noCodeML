import os

import pickle
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# get the working directory of the main.py file 

working_dir= os.path.dirname(os.path.abspath(__file__))

# get the parent directory 

parent_dir= os.path.dirname(working_dir)

def read_data(file_name):
    file_path = f"{parent_dir}/data/{file_name}"
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
        return df
    elif file_path.endswith(".xlsx") or file_path.endswith(".xls"):
        df = pd.read_excel(file_path)
        return df

def preprocess_data(df, target_column, scaler_type):
    X = df.drop(columns=[target_column])
    Y = df[target_column]

    numerical_cols = X.select_dtypes(include = ['number']).columns
    categorical_cols = X.select_dtypes(include = ['object', 'category']).columns

    if len(numerical_cols) == 0:
        pass
    else:
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

        # Impute missing values for numerical columns (mean imputation)
        num_imputer = SimpleImputer(strategy='mean')
        X_train[numerical_cols] = num_imputer.fit_transform(X_train[numerical_cols])
        X_test[numerical_cols] = num_imputer.transform(X_test[numerical_cols])

        # Scale the numerical features based on scaler_type
        if scaler_type == 'Standard':
            scaler = StandardScaler()
        elif scaler_type == 'MinMax':
            scaler = MinMaxScaler()

        X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
        X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

    if len(categorical_cols) == 0:
        pass
    else:
        # Impute missing values for categorical columns (mode imputation)
        cat_imputer = SimpleImputer(strategy='most_frequent')
        X_train[categorical_cols] = cat_imputer.fit_transform(X_train[categorical_cols])
        X_test[categorical_cols] = cat_imputer.transform(X_test[categorical_cols])

        # One-hot encode categorical features
        encoder = OneHotEncoder()
        X_train_encoded = encoder.fit_transform(X_train[categorical_cols])
        X_test_encoded = encoder.transform(X_test[categorical_cols])
        X_train_encoded = pd.DataFrame(X_train_encoded.toarray(), columns=encoder.get_feature_names(categorical_cols))
        X_test_encoded = pd.DataFrame(X_test_encoded.toarray(), columns=encoder.get_feature_names(categorical_cols))
        X_train = pd.concat([X_train.drop(columns=categorical_cols), X_train_encoded], axis=1)
        X_test = pd.concat([X_test.drop(columns=categorical_cols), X_test_encoded], axis=1)

    return X_train, X_test, Y_train, Y_test



def train_model(X_train, Y_train, model, model_name):
    # training the selected model
    model.fit(X_train, Y_train)
    # saving the trained model
    with open(f"{parent_dir}/trainedmodel/{model_name}.pkl", 'wb') as file:
        pickle.dump(model, file)
    return model


def evaluate_model(model, X_test, Y_test):
    Y_pred = model.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred)
    accuracy = round(accuracy, 2)
    return accuracy
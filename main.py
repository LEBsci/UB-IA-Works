'''Import necessary libraries'''
import pandas as pd
import numpy as np
import mlflow, argparse, requests

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.tree import DecisionTreeClassifier, plot_tree


'''Functions to clean and organize the data'''

def question_to_nan(df):
    df = df.replace('?', np.nan)
    return df

# Function to drop non numerical columns except 'income'
def drop_non_numerical(df):
    non_numerical_columns = df.select_dtypes(exclude=[np.number]).columns
    non_numerical_columns = non_numerical_columns.drop('income')
    df = df.drop(columns=non_numerical_columns)
    return df

# Function to change the 'income' column to categorical
def encode_income(df):
    df['income'] = pd.Categorical(df['income'])
    print("Categories and their encoded values:")
    print(dict(enumerate(df['income'].cat.categories)))
    df['income'] = df['income'].cat.codes
    return df

'''Functions to transform the data'''
# Define an imputer for the median of the train data and apply to both train and test
imputer = SimpleImputer(strategy='median')
def impute_data(df):
    df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    return df

'''Define the pipelines'''
# Cleaning and preprocessing
data_preprocessing = Pipeline([
    ('question_to_nan', FunctionTransformer(question_to_nan)),
    ('drop_non_numerical', FunctionTransformer(drop_non_numerical)),
    ('encode_income', FunctionTransformer(encode_income))
])

# Transformations and modifications
data_transformations = Pipeline([
    ('impute_data', FunctionTransformer(impute_data))
])

'''Define the metrics and the model'''

# Function to determine the metrics of the model
def eval_metrics(actual, pred):
    mse = mean_squared_error(actual, pred)
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return mse, mae, r2

def train_model():

    # Download the dataset
    url = "https://raw.githubusercontent.com/pooja2512/Adult-Census-Income/master/adult.csv"
    response = requests.get(url)
    with open("adult.csv", "wb") as f:
        f.write(response.content)

    # Load the dataset
    data = pd.read_csv("adult.csv")
    
    # Preprocess the data
    data_prepared = data_preprocessing.fit_transform(data)
    
    # Divide the data into test and train with stratify on the target
    train_data, test_data = train_test_split(data_prepared, test_size=0.2, stratify=data_prepared['income'], random_state=42)

    # Define the features and the target
    X_train = train_data.drop(columns='income')
    y_train = train_data['income']
    
    X_test = test_data.drop(columns='income')
    y_test = test_data['income']

    # Transform the data
    X_train = data_transformations.fit_transform(X_train)
    X_test = data_transformations.transform(X_test)

    # Create a Decision Tree model
    model = DecisionTreeClassifier()

    # Fit the model
    model.fit(X_train, y_train)

    # Log the model
    mlflow.sklearn.log_model(model, "DecisionTreeModel", input_example=X_train.head(50))

    # Predict the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse, mae, r2 = eval_metrics(y_test, y_pred)

    # Log the metrics
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)



if __name__ == "__main__":
    train_model()
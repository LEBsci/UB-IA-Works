# Import necessary libraries
import pandas as pd
import mlflow, argparse

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import ConfusionMatrixDisplay

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def eval_metrics(actual, pred):
    mse = mean_squared_error(actual, pred)
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return mse, mae, r2

def train_model(max_iter=200):

    # Load the Iris dataset
    iris = load_iris()
    
    # Transform the dataset into a DataFrame
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
    
    # Split the dataset into training and testing sets
    x, xt, y, yt = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)
    # Start to log an experiment

    # print(f"Starting the experiment with max_iter={max_iter}")

    # Log the parameters
    mlflow.log_param("max_iter", max_iter)

    # Create a Logistic Regression model
    model = LogisticRegression(max_iter=max_iter)

    # Fit the model
    model.fit(x, y)

    # Log the model
    mlflow.sklearn.log_model(model, "model")

    # Predict the test set
    y_pred = model.predict(xt)

    # Check the metrics
    mse, mae, r2 = eval_metrics(yt, y_pred)

    # Log the metrics
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_iter", type=int, default=200)
    args = parser.parse_args()
    train_model(args.max_iter)
import mlflow 
from mlflow.models import infer_signature

import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# ##
# Load the Iris dataset
X, y = datasets.load_iris(return_X_y=True)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define the parameter grid for GridSearchCV
param_grid = {
    "solver": ["newton-cg", "lbfgs", "sag", "saga"],
    "max_iter": [100, 500, 1000],
    "multi_class": ["auto", "ovr", "multinomial"],
    "C": [0.1, 1.0, 10.0],  # Inverse of regularization strength
    "penalty": ["l2"],       # LogisticRegression default penalty
    "random_state": [8888],
}

# Initialize LogisticRegression (without setting hyperparameters)
lr = LogisticRegression()

# Initialize GridSearchCV
grid_search = GridSearchCV(
    estimator=lr,
    param_grid=param_grid,
    cv=5,                     # 5-fold cross-validation
    scoring="accuracy",       # Evaluation metric
    n_jobs=-1,                # Use all available cores
    verbose=1,                # Verbosity mode
)

# Fit GridSearchCV to find the best hyperparameters
grid_search.fit(X_train, y_train)

# Retrieve the best estimator
best_lr = grid_search.best_estimator_

# Predict on the test set using the best estimator
y_pred = best_lr.predict(X_test)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted")
recall = recall_score(y_test, y_pred, average="weighted")
f1 = f1_score(y_test, y_pred, average="weighted")

# Set MLflow tracking server URI for logging
mlflow.set_tracking_uri(uri="http://127.0.0.1:5001")

# Create or set an MLflow Experiment
mlflow.set_experiment("Iris LR GridSearchCV")

# Start an MLflow run
with mlflow.start_run():
    # Log the best hyperparameters found by GridSearchCV
    mlflow.log_params(grid_search.best_params_)
    
    # Log evaluation metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    
    # Set a descriptive tag for the run
    mlflow.set_tag("Training Info", "Logistic Regression with GridSearchCV on Iris data")
    
    # Infer the model signature (input and output schema)
    signature = infer_signature(X_train, best_lr.predict(X_train))
    
    # Log the best model to MLflow
    model_info = mlflow.sklearn.log_model(
        sk_model=best_lr,
        artifact_path="iris_model",
        signature=signature,
        input_example=X_train[:5],  # Example inputs for the model
        registered_model_name="iris_model",
    )
    
    print("Best Model Parameters:", grid_search.best_params_)
    print("Test Set Metrics:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")
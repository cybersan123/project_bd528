from mlflow.models import validate_serving_input

model_uri = '/Users/macbookpro/Library/CloudStorage/OneDrive-Personal/Doc/MLflow_project/mlflow/examples/depression/mlruns/1/0fec74a519044a019d814836bf95eee0/artifacts/model/MLmodel'

# The logged model does not contain an input_example.
# Manually generate a serving payload to verify your model prior to deployment.
from mlflow.models import convert_input_example_to_serving_input

# Define INPUT_EXAMPLE via assignment with your own input example to the model
# A valid input example is a data instance suitable for pyfunc prediction
serving_payload = convert_input_example_to_serving_input(input_example=X_train[:5])

# Validate the serving payload works on the model
validate_serving_input(model_uri, serving_payload)
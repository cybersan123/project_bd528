# tensorflow 2.x core api
import tensorflow as tf
from sklearn.datasets import fetch_california_housing

import mlflow
from mlflow.models import infer_signature
import pandas as pd  # Added import for pandas

class Normalize(tf.Module):
    """Data Normalization class"""

    def __init__(self, x):
        super().__init__()
        # Initialize the mean and standard deviation for normalization
        self.mean = tf.math.reduce_mean(x, axis=0)
        self.std = tf.math.reduce_std(x, axis=0)

    @tf.function
    def norm(self, x):
        return (x - self.mean) / self.std

    @tf.function
    def unnorm(self, x):
        return (x * self.std) + self.mean

class MLP(tf.keras.Model):
    """Multi-Layer Perceptron model class using tf.keras.Model"""

    def __init__(self, hidden_units=[64, 32], activation='relu'):
        super(MLP, self).__init__()
        # Define hidden layers
        self.hidden_layers = []
        for units in hidden_units:
            self.hidden_layers.append(tf.keras.layers.Dense(units, activation=activation))
        # Output layer for regression
        self.output_layer = tf.keras.layers.Dense(1)  # Outputs a single value

    def call(self, inputs):
        x = inputs
        for layer in self.hidden_layers:
            x = layer(x)
        y = self.output_layer(x)
        return tf.squeeze(y, axis=1)  # Remove the last dimension

class ExportModule(tf.Module):
    """Exporting TF model"""

    def __init__(self, model, norm_x, norm_y):
        super().__init__()
        # Initialize pre and postprocessing functions
        self.model = model
        self.norm_x = norm_x
        self.norm_y = norm_y

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 8], dtype=tf.float32)])
    def __call__(self, x):
        # Run the ExportModule for new data points
        x = self.norm_x.norm(x)
        y = self.model(x)
        y = self.norm_y.unnorm(y)
        return y

def mse_loss(y_true, y_pred):
    """Calculating Mean Square Error Loss function"""
    return tf.reduce_mean(tf.square(y_pred - y_true))

if __name__ == "__main__":
    # Set a random seed for reproducible results
    tf.random.set_seed(42)

    # Load dataset
    housing = fetch_california_housing(as_frame=True)
    dataset = housing["frame"]
    feature_names = housing["feature_names"]  # Extract feature names
    # Drop missing values
    dataset = dataset.dropna()
    # Using only first 1500 samples
    dataset = dataset[:1500]
    dataset_tf = tf.convert_to_tensor(dataset, dtype=tf.float32)

    # Split dataset into train and test
    dataset_shuffled = tf.random.shuffle(dataset_tf, seed=42)
    train_data, test_data = dataset_shuffled[100:], dataset_shuffled[:100]
    x_train, y_train = train_data[:, :-1], train_data[:, -1]
    x_test, y_test = test_data[:, :-1], test_data[:, -1]
    # Data normalization
    norm_x = Normalize(x_train)
    norm_y = Normalize(y_train)
    x_train_norm, y_train_norm = norm_x.norm(x_train), norm_y.norm(y_train)
    x_test_norm, y_test_norm = norm_x.norm(x_test), norm_y.norm(y_test)

    # Create input_example before normalization
    # Convert first 5 samples from x_test to a pandas DataFrame with feature names
    input_example_raw = x_test[:5].numpy()  # Shape: [5, 8], dtype: float32
    input_example = pd.DataFrame(input_example_raw, columns=feature_names).astype('float32')  # Ensure dtype is float32

    # Set MLflow tracking URI and experiment
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("TF_Example MLP")

    with mlflow.start_run():
        # Initialize MLP model
        mlp = MLP(hidden_units=[64, 32], activation='relu')

        # Compile the model
        mlp.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                    loss=mse_loss,
                    metrics=['mse'])

        # Prepare training and testing datasets
        batch_size = 32
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train_norm, y_train_norm))
        train_dataset = train_dataset.shuffle(buffer_size=x_train.shape[0], seed=42).batch(batch_size)
        test_dataset = tf.data.Dataset.from_tensor_slices((x_test_norm, y_test_norm))
        test_dataset = test_dataset.batch(batch_size)

        # Train the model using Keras's fit method
        print("Training model...")
        history = mlp.fit(
            train_dataset,
            epochs=100,
            validation_data=test_dataset,
            verbose=0,  # Set to 1 to see progress
        )

        # Extract final training and testing loss
        final_train_loss = history.history['loss'][-1]
        final_test_loss = history.history['val_loss'][-1]

        # Log the parameters
        mlflow.log_params(
            {
                "epochs": 100,
                "learning_rate": 0.01,
                "batch_size": batch_size,
                "hidden_units": [64, 32],
                "activation": "relu",
                "optimizer": "Adam",
            }
        )

        # Log the final metrics
        mlflow.log_metrics(
            {
                "final_train_loss": final_train_loss,
                "final_test_loss": final_test_loss,
            }
        )
        print(f"\nFinal train loss: {final_train_loss:.3f}")
        print(f"Final test loss: {final_test_loss:.3f}")

        # Export the tensorflow model
        # Wrap the Keras model with ExportModule for normalization
        mlp_export = ExportModule(model=mlp, norm_x=norm_x, norm_y=norm_y)

        # Infer model signature using the raw input and predictions
        predictions = mlp_export(x_test[:5])
        signature = infer_signature(input_example_raw, predictions.numpy())

        # Log the model with input_example
        mlflow.tensorflow.log_model(
            mlp_export,
            "model",
            signature=signature,
            input_example=input_example,  # Added input_example
            registered_model_name="california_housing_mlp_model",  # Optional: Register the model
        )

        print("Model logged to MLflow with input_example.")

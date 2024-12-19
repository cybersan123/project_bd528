import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout, Embedding, Input
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import mlflow
from mlflow.models import infer_signature
import mlflow.keras

# Suppress TensorFlow warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
SEED = 10

# Set MLflow tracking URI and experiment
mlflow.set_tracking_uri("http://127.0.0.1:5001")  # Replace with your server's tracking URI
mlflow.set_experiment("Depression Detection")

# Load dataset
url = 'https://drive.google.com/uc?export=download&id=19-O6-b7PLX3uHmfoLAvVdO0OqF8xsn9v'
df = pd.read_csv(url)

# Prepare data
X = df['post_text']
y = df['label']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

# Visualize the target distribution
plt.figure(figsize=(10, 6))
sns.countplot(x=df['label'], palette='Set1', alpha=0.8)
plt.title('Distribution of Target')
plt.show()

# Visualize word count distribution
df['num_words'] = df['post_text'].apply(lambda x: len(x.split()))
plt.figure(figsize=(20, 6))
sns.histplot(df['num_words'], bins=range(1, 40, 2), palette='Set1', alpha=0.8)
plt.title('Distribution of the Word Count')
plt.show()

# Tokenization and sequence preparation
tok = Tokenizer()
tok.fit_on_texts(X_train)

sequences = tok.texts_to_sequences(X_train)
test_sequences = tok.texts_to_sequences(X_test)

MAX_LEN = 40
X_train_seq = pad_sequences(sequences, maxlen=MAX_LEN)
X_test_seq = pad_sequences(test_sequences, maxlen=MAX_LEN)

vocab_size = len(tok.word_index) + 1

# Model definition
model = Sequential([
    Input(name='inputs', shape=[MAX_LEN]),
    Embedding(vocab_size, 128),
    Bidirectional(LSTM(64, return_sequences=True)),
    Bidirectional(LSTM(32)),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])

# MLflow tracking
with mlflow.start_run():
    # Log parameters
    mlflow.log_param("max_len", MAX_LEN)
    mlflow.log_param("vocab_size", vocab_size)
    mlflow.log_param("batch_size", 64)
    mlflow.log_param("embedding_dim", 128)
    mlflow.log_param("lstm_units", [64, 32])
    
    # Train the model
    history = model.fit(X_train_seq, y_train, epochs=5, validation_split=0.2, batch_size=64,
                        callbacks=[EarlyStopping(monitor='val_accuracy', mode='max', patience=3,
                                                 verbose=False, restore_best_weights=True)])
    
    # Log metrics for training and validation
    for epoch in range(len(history.history['accuracy'])):
        mlflow.log_metric("train_accuracy", history.history['accuracy'][epoch], step=epoch)
        mlflow.log_metric("val_accuracy", history.history['val_accuracy'][epoch], step=epoch)
        mlflow.log_metric("train_loss", history.history['loss'][epoch], step=epoch)
        mlflow.log_metric("val_loss", history.history['val_loss'][epoch], step=epoch)
    
    # Evaluate the model
    test_loss, test_acc = model.evaluate(X_test_seq, y_test)
    y_hat = model.predict(X_test_seq)
    
    # Log final test metrics
    mlflow.log_metric("test_loss", test_loss)
    mlflow.log_metric("test_accuracy", test_acc)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

     # Set a descriptive tag for the run
    mlflow.set_tag("Training Info", "Depression")

    signature = infer_signature(X_train, model.predict(X_train))
    
    # Log classification report as an artifact
    report = classification_report(y_test, np.where(y_hat >= 0.5, 1, 0), output_dict=True)
    mlflow.log_dict(report, "classification_report.json")
    
    # Log the model
    mlflow.keras.log_model(
        sk_model= model, 
        artifact_path="model",
        signature=signature,
        input_example=X_train[:5],  # Example inputs for the model
        registered_model_name="model",
    )
    
    # Log confusion matrix as an artifact
    cm = confusion_matrix(y_test, np.where(y_hat >= 0.5, 1, 0))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='', cmap='Blues')
    plt.xlabel('Predicted Labels')
    plt.ylabel('Real Labels')
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")

print('Test Loss:', test_loss)
print('Test Accuracy:', test_acc)
print("accuracy", accuracy)
print ("precision", precision)
print("recall", recall)
print("f1_score", f1)
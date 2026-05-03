import numpy as np
import time
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, LSTM, GRU, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences

vocab_size = 10000  # top 10k words
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size)

max_len = 200
X_train = pad_sequences(X_train, maxlen=max_len)
X_test = pad_sequences(X_test, maxlen=max_len)

def build_model(model_type):
    model = Sequential()

    # Embedding layer
    model.add(Embedding(input_dim=vocab_size, output_dim=128, input_length=max_len))

    # Choose model
    if model_type == "RNN":
        model.add(SimpleRNN(64))

    elif model_type == "LSTM":
        model.add(LSTM(64))

    elif model_type == "GRU":
        model.add(GRU(64))

    # Output layer
    model.add(Dense(1, activation='sigmoid'))

    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    return model

results = {}

for model_type in ["RNN", "LSTM", "GRU"]:
    print(f"\nTraining {model_type}...")

    model = build_model(model_type)

    start_time = time.time()

    history = model.fit(
        X_train, y_train,
        epochs=3,
        batch_size=64,
        validation_split=0.2,
        verbose=1
    )

    end_time = time.time()

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

    results[model_type] = {
        "history": history,
        "loss": loss,
        "accuracy": accuracy,
        "time": end_time - start_time
    }

print("\n Final Comparison:")

for model_type in results:
    print(f"{model_type}:")
    print(f"Accuracy: {results[model_type]['accuracy']:.4f}")
    print(f"Loss: {results[model_type]['loss']:.4f}")
    print(f"Training Time: {results[model_type]['time']:.2f} sec\n")

for model_type in results:
    history = results[model_type]["history"]

    plt.plot(history.history['accuracy'], label=f"{model_type} Train")
    plt.plot(history.history['val_accuracy'], linestyle='--', label=f"{model_type} Val")

plt.title("Accuracy Comparison")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

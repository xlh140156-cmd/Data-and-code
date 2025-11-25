import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.models import Model

def load_atsd():
    data = np.load("atsd_components.npz")
    # Demo: combine a few components
    series = data["A1"] + data["D1"] + data["D2"]
    return series.reshape(-1, 1)

def build_demo_model():
    inp = Input(shape=(12, 1))
    x = LSTM(32, return_sequences=False)(inp)
    out = Dense(1)(x)
    return Model(inp, out)

def create_dataset(series, n=12):
    X, y = [], []
    for i in range(len(series)-n):
        X.append(series[i:i+n])
        y.append(series[i+n])
    return np.array(X), np.array(y)

if __name__ == "__main__":
    series = load_atsd()
    X, y = create_dataset(series, 12)

    model = build_demo_model()
    model.compile(optimizer="adam", loss="mse")

    model.fit(X, y, epochs=10, batch_size=32, verbose=0)

    pred = model.predict(X, verbose=0)

    np.save("y_true.npy", y)
    np.save("y_pred.npy", pred)

    print("Saved y_true.npy, y_pred.npy")

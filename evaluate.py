import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def mase(true, pred):
    diff = np.abs(np.diff(true.flatten()))
    d = np.mean(diff) + 1e-8
    return np.mean(np.abs(true - pred)) / d

if __name__ == "__main__":
    y_true = np.load("y_true.npy").flatten()
    y_pred = np.load("y_pred.npy").flatten()

    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    mase_v = mase(y_true, y_pred)

    df = pd.DataFrame({
        "Metric": ["MAE", "RMSE", "R2", "MASE"],
        "Value": [mae, rmse, r2, mase_v]
    })

    df.to_csv("metrics.csv", index=False)
    print(df)

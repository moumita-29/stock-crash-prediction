import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os

SEQ_LEN = 30

def create_sequences():

    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # load features
    df = pd.read_csv("data/processed/features.csv")

    # keep numeric columns
    numeric_df = df.select_dtypes(include=['number'])

    # split features and labels
    drop_cols = []

    if 'crash' in numeric_df.columns:
        drop_cols.append('crash')

    if 'future_return' in numeric_df.columns:
        drop_cols.append('future_return')

    features = numeric_df.drop(drop_cols, axis=1).values
    labels = numeric_df['crash'].values

    # scale features
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # save scaler
    joblib.dump(scaler, "models/scaler.pkl")

    X = []
    y = []

    for i in range(len(features) - SEQ_LEN):
        X.append(features[i:i+SEQ_LEN])
        y.append(labels[i+SEQ_LEN])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    np.save("data/processed/X.npy", X)
    np.save("data/processed/y.npy", y)

    print("Dataset created successfully")
    print("X shape:", X.shape)
    print("y shape:", y.shape)


if __name__ == "__main__":
    create_sequences()
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


def load_data(path):
    return pd.read_csv(path)


def scale_features(df):
    scaler = StandardScaler()
    df["Time"] = scaler.fit_transform(df[["Time"]])
    df["Amount"] = scaler.fit_transform(df[["Amount"]])
    return df, scaler


def split_data(df):
    X, y = df.drop(columns=["Class"]), df["Class"]
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    return X_train, X_val, X_test, y_train, y_val, y_test


def apply_smote(X_train, y_train):
    return SMOTE(random_state=42).fit_resample(X_train, y_train)


def create_undersample(X_train, y_train):
    df = pd.concat([X_train, y_train], axis=1)
    legit, fraud = df[df["Class"] == 0], df[df["Class"] == 1]
    legit_under = legit.sample(n=len(fraud), random_state=42)
    under = pd.concat([legit_under, fraud])
    return under.drop(columns=["Class"]), under["Class"]

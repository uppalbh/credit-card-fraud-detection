from sklearn.linear_model import LogisticRegression


def train_baseline(X_train, y_train):
    m = LogisticRegression(max_iter=1000, random_state=42)
    m.fit(X_train, y_train)
    return m


def train_undersampled(X_train, y_train):
    m = LogisticRegression(max_iter=1000, random_state=42)
    m.fit(X_train, y_train)
    return m


def train_smote(X_train, y_train):
    m = LogisticRegression(max_iter=1000, random_state=42)
    m.fit(X_train, y_train)
    return m

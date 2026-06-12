from sklearn.metrics import accuracy_score, classification_report, roc_curve, roc_auc_score


def evaluate_model(name, model, X_val, y_val, X_test, y_test):
    val_pred, test_pred = model.predict(X_val), model.predict(X_test)

    print("\n===", name, "===")
    print("Val Acc:", accuracy_score(y_val, val_pred))
    print(classification_report(y_val, val_pred))
    print("Test Acc:", accuracy_score(y_test, test_pred))
    print(classification_report(y_test, test_pred))

    return val_pred, test_pred


def compute_roc(model, X_test, y_test):
    p = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, p)
    return fpr, tpr, roc_auc_score(y_test, p)

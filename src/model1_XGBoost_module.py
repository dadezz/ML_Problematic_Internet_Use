from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import optuna
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from data_cleaning_module import season_map
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import logging

def tune_and_train_xgboost(X_train, y_train, X_val, y_val, num_classes, n_trials=50):

    class_counts = np.bincount(y_train)
    total = len(y_train)
    weights = {i: total / (len(class_counts) * c) for i, c in enumerate(class_counts)}
    sample_weights = np.array([weights[label] for label in y_train])

    def objective(trial):
        params = {
            'objective': 'multi:softprob',
            'num_class': num_classes,
            'tree_method': 'hist',
            'eval_metric': ['mlogloss', 'merror'],
            'max_depth': trial.suggest_int('max_depth', 2, 6),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.05, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),
        }

        model = XGBClassifier(**params, n_estimators=1000, verbosity=0, early_stopping_rounds=50)
        model.fit(X_train, y_train, sample_weight=sample_weights,
                  eval_set=[(X_val, y_val)],verbose=False)
        y_pred_val = model.predict(X_val)
        return f1_score(y_val, y_pred_val, average='weighted')
    
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    print("Best hyperparameters:", study.best_params)

    best_params = study.best_params
    best_params.update({
        'objective': 'multi:softprob',
        'num_class': num_classes,
        'tree_method': 'auto',
        'eval_metric': ['mlogloss', 'merror']
    })

    model_final = XGBClassifier(**best_params, n_estimators=1000, verbosity=0,early_stopping_rounds=50)
    model_final.fit(X_train, y_train, sample_weight=sample_weights,
                    eval_set=[(X_train, y_train),(X_val, y_val)],verbose=False)
    
    model_final = XGBClassifier(**best_params, n_estimators=1000, verbosity=0,early_stopping_rounds=50)
    model_final.fit(X_train, y_train, sample_weight=sample_weights,
                    eval_set=[(X_train, y_train),(X_val, y_val)],verbose=False)

    evals_result = model_final.evals_result()

    epochs = range(len(evals_result['validation_0']['mlogloss']))
    fig, axes = plt.subplots(1, 3, figsize=(20,5))

    axes[0].plot(epochs, evals_result['validation_0']['mlogloss'], label='Train Loss')
    axes[0].plot(epochs, evals_result['validation_1']['mlogloss'], label='Validation Loss')
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Log Loss")
    axes[0].set_title("Training vs Validation Loss")
    axes[0].legend()
    axes[0].grid(True)

    train_acc = [1 - x for x in evals_result['validation_0']['merror']]
    val_acc = [1 - x for x in evals_result['validation_1']['merror']]
    axes[1].plot(epochs, train_acc, label='Train Accuracy')
    axes[1].plot(epochs, val_acc, label='Validation Accuracy')
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Training vs Validation Accuracy")
    axes[1].legend()
    axes[1].grid(True)

    y_pred_val = model_final.predict(X_val)
    cm_val = confusion_matrix(y_val, y_pred_val)
    sns.heatmap(cm_val, annot=True, fmt="d", cmap="Greens", cbar=False, ax=axes[2])
    axes[2].set_title("Confusion Matrix - Validation")
    axes[2].set_xlabel("Predicted")
    axes[2].set_ylabel("True")
    

    print("=== Validation Set ===")
    print("Accuracy:", accuracy_score(y_val, y_pred_val))
    print("Weighted F1:", f1_score(y_val, y_pred_val, average='weighted'))

    # Classification report come tabella grafica
    report_dict = classification_report(y_val, y_pred_val, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose().round(2)

    plt.figure(figsize=(10, report_df.shape[0]*0.6))
    sns.heatmap(report_df.iloc[:-1, :-1], annot=True, cmap="Blues", cbar=False, fmt=".2f")
    plt.title("Classification Report - Validation Set")
    plt.ylabel("Classes")
    plt.xlabel("Metrics")

    plt.tight_layout()
    plt.show()

    return model_final, study

def test_xgboost(model, X_test, y_test, classes=["0","1","2","3"], SEED=42):

    y_pred = model.predict(X_test)
    print("=== XGBoost - Classification Report (Test) ===")
    print(classification_report(y_test, y_pred, digits=4))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("XGBoost Confusion Matrix (Test)")
    plt.show()


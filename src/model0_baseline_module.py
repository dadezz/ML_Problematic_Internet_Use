from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def run_baseline(X_train_final, y_train_final, X_test, y_test):
    y_train_mode = y_train_final.mode()[0]

    # Predizione costante sul test
    y_pred_base = [y_train_mode] * len(y_test)

    # Accuracy baseline
    base_accuracy = accuracy_score(y_test, y_pred_base)
    print("Baseline accuracy:", base_accuracy)

    cm = confusion_matrix(y_test, y_pred_base)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=sorted(set(y_test)))
    disp.plot(cmap=plt.cm.Blues)
    plt.show()
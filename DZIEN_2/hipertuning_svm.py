import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# 1) Dane (syntetyczne, ale realistyczne do klasyfikacji)
X, y = make_classification(
    n_samples=5000,
    n_features=20,
    n_informative=12,
    n_redundant=6,
    n_classes=2,
    class_sep=1.2,
    flip_y=0.03,        # trochę szumu w etykietach
    random_state=42
)

# 2) Podział train/test (stratyfikacja, żeby proporcje klas były zachowane)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    stratify=y,
    random_state=42
)

# 3) Pipeline: skalowanie + SVM
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC())
])

# 4) Siatka hiperparametrów (RBF + porównawczo linear)
param_grid = [
    {   # SVM z jądrem liniowym
        "svm__kernel": ["linear"],
        "svm__C": [0.1, 1, 10, 100],
    },
    {   # SVM z RBF
        "svm__kernel": ["rbf"],
        "svm__C": [0.1, 1, 10, 100],
        "svm__gamma": ["scale", 0.1, 0.01, 0.001],
    }
]

# 5) Strojenie (CV=5)
grid = GridSearchCV(
    pipe,
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

grid.fit(X_train, y_train)

print("Najlepsze parametry:", grid.best_params_)
print("Najlepsza accuracy (CV):", grid.best_score_)

# 6) Ocena na zbiorze testowym
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)

test_acc = accuracy_score(y_test, y_pred)
print("Accuracy (TEST):", test_acc)

print("\nRaport klasyfikacji:")
print(classification_report(y_test, y_pred))

# 7) (Opcjonalnie) Macierz pomyłek
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.title("Macierz pomyłek (SVM po tuningu)")
plt.show()

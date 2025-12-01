import time 
import numpy as np
import pandas as pd
import pyreadstat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from qiskit.circuit.library import ZZFeatureMap
from qiskit.circuit.library import RealAmplitudes
from qiskit_machine_learning.algorithms.classifiers import VQC
from qiskit.circuit.library import EfficientSU2, ExcitationPreserving
from qiskit_machine_learning.optimizers import COBYLA
from qiskit.primitives import StatevectorSampler as Sampler
from itertools import product
from sklearn.metrics import recall_score, precision_score, f1_score

start_time = time.time()
print("Reading in data.")
df, meta = pyreadstat.read_sav("romantic_love_survey_data.sav")

df_narrowed = df.iloc[:,  24:-1].dropna()
X = df_narrowed.drop("Satisfaction_1", axis=1)
y = df_narrowed['Satisfaction_1']
y = ((y > 3).astype(int)).to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
scaled_data = scaler.fit(X_train)

X_trains = scaler.transform(X_train)
X_tests = scaler.transform(X_test)
print("Dimensionality Reduction.")
pca = PCA(n_components=8)
pca_fitted = pca.fit(X_trains)

X_trains_components = pca.transform(X_trains)
X_tests_components = pca.transform(X_tests)

num_features = X_trains_components.shape[1]
print("Starting quantum training.")
feature_map = ZZFeatureMap(feature_dimension=num_features, reps=1)

def kfold_indices(data, k):
    fold_size = len(data) // k
    indices = np.arange(len(data))
    folds = []
    for i in range(k):
        test_indices = indices[i * fold_size: (i + 1) * fold_size]
        train_indices = np.concatenate([indices[:i * fold_size], indices[(i + 1) * fold_size:]])
        folds.append((train_indices, test_indices))
    return folds

k = 3
fold_indices = kfold_indices(X_trains_components, k)

scores = pd.DataFrame(columns=["CV Time", "Train Score", "Validation Score", "ansatz", "optimizer", "reps"])

parameter_grid = {
    "ansatz": ["RealAmplitudes", "EfficientSU2", "ExcitationPreserving"],
    "optimizer": ["COBYLA"],
    "reps": [1, 2, 3, 4, 5, 6]
    }

param_dict = {"RealAmplitudes": RealAmplitudes, "EfficientSU2": EfficientSU2, "ExcitationPreserving": ExcitationPreserving, "COBYLA": COBYLA(maxiter=40)}
param_names = parameter_grid.keys()
param_values = parameter_grid.values()

for combination_values in product(*param_values):
    cv_time_start = time.time()
    current_params = dict(zip(param_names, combination_values))
    print(current_params)
    current_model_train_scores = []
    current_model_val_scores = []
    for train_indices, val_indices in fold_indices:
        X_train_split, y_train_split = X_trains_components[train_indices], y_train[train_indices]
        X_val, y_val = X_trains_components[val_indices], y_train[val_indices]
        vqc = VQC(
        sampler=Sampler(),
        feature_map=feature_map,
        ansatz=param_dict[current_params["ansatz"]](num_qubits=num_features, reps=current_params['reps']),
        optimizer=param_dict[current_params['optimizer']],
    )
        print("Training model")
        vqc.fit(X_train_split, y_train_split)
        
        train_score = vqc.score(X_train_split, y_train_split)
        
        fold_score = vqc.score(X_val, y_val)
        
        current_model_train_scores.append(train_score)
        current_model_val_scores.append(fold_score)
    cv_time_end = time.time()
    elapsed_cv_time = cv_time_end - cv_time_start
    mean_accuracy_train = np.mean(current_model_train_scores)
    mean_accuracy_val = np.mean(current_model_val_scores)
    new_row = pd.DataFrame([[elapsed_cv_time, mean_accuracy_train, mean_accuracy_val, current_params['ansatz'], current_params['optimizer'], current_params['reps']]], columns=["CV Time", "Train Score", "Validation Score", "ansatz", "optimizer", "reps"])
    scores = pd.concat([scores, new_row], ignore_index=True)
    print(scores)

scores.to_csv("vqc_cv_results.csv", index=False)
# Find best model
scores.sort_values(by="Validation Score", ascending=False, inplace=True)
scores.reset_index(drop=True, inplace=True)

print("Best score on validation set:", scores.iloc[0]['Validation Score'])

# Train the best model
best_model_start_time = time.time()

best_vqc = VQC(
sampler=Sampler(),
feature_map=feature_map,
ansatz=param_dict[scores.iat[0,3]](num_qubits=num_features, reps=scores.iat[0,5]),
optimizer=param_dict[scores.iat[0,4]],
)
print("Training model")
best_vqc.fit(X_trains_components, y_train)
best_model_end_time = time.time()
best_model_elapsed_time = best_model_end_time - best_model_start_time
train_score = best_vqc.score(X_trains_components, y_train)
test_score = best_vqc.score(X_tests_components, y_test)

test_predictions = best_vqc.predict(X_tests_components)
recall = recall_score(y_test, test_predictions)
precision = precision_score(y_test, test_predictions)
f1 = f1_score(y_test, test_predictions)
final_columns = ["TrainTime", "ansatz", "optimizer", "reps", "TrainAccuracy", "Accuracy", "Recall", "Precision", "F1 Score"]

final_results = pd.DataFrame([[best_model_elapsed_time, scores.iat[0,3], scores.iat[0,4], scores.iat[0,5], train_score, test_score, recall, precision, f1]], columns=final_columns)
final_results.to_csv("best_vqc_results.csv", index=False)
import time 
import pandas as pd
import pyreadstat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import recall_score, precision_score, f1_score

start_time = time.time()

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

pca = PCA(n_components=8)
pca_fitted = pca.fit(X_trains)

X_trains_components = pca.transform(X_trains)
X_tests_components = pca.transform(X_tests)


param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'poly'],
        'gamma': [0.01, 0.1, 1]
    }

svc = SVC(random_state=42)

grid_search = GridSearchCV(svc, param_grid, cv=3, verbose=2) 

# Fitting best model
grid_search.fit(X_trains_components, y_train)
best_model_start = time.time()

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_}")


best_svm = grid_search.best_estimator_
test_accuracy = best_svm.score(X_tests_components, y_test)
print(f"Test set accuracy: {test_accuracy}")

train_score = best_svm.score(X_trains_components, y_train)
test_score = best_svm.score(X_tests_components, y_test)

test_predictions = best_svm.predict(X_tests_components)
recall = recall_score(y_test, test_predictions)
precision = precision_score(y_test, test_predictions)
f1 = f1_score(y_test, test_predictions)

print(f"Classical SVC on the training dataset: {train_score:.2f}")
print(f"Classical SVC on the test dataset:     {test_score:.2f}")
best_model_end = time.time()
best_model_elapsed_time = best_model_end - best_model_start
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time to train SVC: {elapsed_time:.4f} seconds")

final_columns = ["TrainTime", "C", "kernel", "gamma", "TrainAccuracy", "Accuracy", "Recall", "Precision", "F1 Score"]

final_results = pd.DataFrame([[best_model_elapsed_time, grid_search.best_params_["C"], grid_search.best_params_["kernel"], grid_search.best_params_["gamma"], train_score, test_score, recall, precision, f1]], columns=final_columns)
final_results.to_csv("best_svm_results.csv", index=False)
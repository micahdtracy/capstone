import time 
import pandas as pd
import pyreadstat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

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
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': [0.001, 0.01, 0.1, 1, 'scale', 'auto']
    }

svc = SVC(random_state=42)

grid_search = GridSearchCV(svc, param_grid, cv=3, verbose=2, n_jobs=-1) # n_jobs=-1 uses all available cores
grid_search.fit(X_trains_components, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_}")


best_svm = grid_search.best_estimator_
test_accuracy = best_svm.score(X_tests_components, y_test)
print(f"Test set accuracy: {test_accuracy}")

train_score_c4 = best_svm.score(X_trains_components, y_train)
test_score_c4 = best_svm.score(X_tests_components, y_test)

print(f"Classical SVC on the training dataset: {train_score_c4:.2f}")
print(f"Classical SVC on the test dataset:     {test_score_c4:.2f}")

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time to train SVC: {elapsed_time:.4f} seconds")


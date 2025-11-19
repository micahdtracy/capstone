import time 
import pandas as pd
import pyreadstat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from qiskit.circuit.library import ZZFeatureMap
from qiskit.circuit.library import RealAmplitudes
from qiskit_machine_learning.algorithms.classifiers import VQC
from qiskit.circuit.library import EfficientSU2
from qiskit_machine_learning.optimizers import COBYLA
from qiskit.primitives import StatevectorSampler as Sampler


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
print("Starting quantum stuff: RealAmplitudes")
feature_map = ZZFeatureMap(feature_dimension=num_features, reps=1)

for i in range(3, 11):
    ansatz = RealAmplitudes(num_qubits=num_features, reps=i)

    # maybe try some different optimizers? different numbers of layers?
    # Maybe look here to figure out how to tune hyperparameters?
    # https://github.com/qiskit-community/qiskit-machine-learning/issues/494
    # https://quantumcomputing.stackexchange.com/questions/21574/how-to-set-hyper-parameters-for-a-variational-quantum-classifier-qiskit
    # https://arxiv.org/pdf/2405.12354
    # https://www.sciencedirect.com/science/article/pii/S2950257824000076
    # https://aws.amazon.com/blogs/quantum-computing/tracking-quantum-experiments-with-amazon-braket-hybrid-jobs-and-amazon-sagemaker-experiments/
    optimizer = COBYLA(maxiter=100)

    sampler = Sampler()


    vqc = VQC(
        sampler=sampler,
        feature_map=feature_map,
        ansatz=ansatz,
        optimizer=optimizer,
    )

    # clear objective value history
    objective_func_vals = []

    start = time.time()
    print("Starting training")
    vqc.fit(X_trains_components, y_train)
    elapsed = time.time() - start

    print(f"Training time for {i} reps: {round(elapsed)} seconds")

    train_score_q4 = vqc.score(X_trains_components, y_train)
    test_score_q4 = vqc.score(X_tests_components, y_test)

    print(f"Quantum VQC RealAmplitudes, {i} reps on the training dataset: {train_score_q4:.2f}")
    print(f"Quantum VQC RealAmplitudes, {i} reps on the test dataset:     {test_score_q4:.2f}")

print("Training EfficientSU2")
for i in range(3, 11):

    # change reps, change type of ansatz
    ansatz = EfficientSU2(num_qubits=num_features, reps=i)
    optimizer = COBYLA(maxiter=40)

    vqc = VQC(
        sampler=sampler,
        feature_map=feature_map,
        ansatz=ansatz,
        optimizer=optimizer,
    )

    # clear objective value history
    objective_func_vals = []

    start = time.time()
    print("Starting training")
    vqc.fit(X_trains_components, y_train)
    elapsed = time.time() - start

    print(f"Training time for {i} reps: {round(elapsed)} seconds")

    train_score_q2_eff = vqc.score(X_trains_components, y_train)
    test_score_q2_eff = vqc.score(X_tests_components, y_test)

    print(f"Quantum VQC on the training dataset using EfficientSU2 with {i} reps: {train_score_q2_eff:.2f}")
    print(f"Quantum VQC on the test dataset using EfficientSU2 with {i} reps:     {test_score_q2_eff:.2f}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time to train VQC: {elapsed_time:.4f} seconds")





from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from qiskit import Aer
from qiskit.aqua.components.feature_maps import SecondOrderExpansion,FirstOrderExpansion
from qiskit.aqua.algorithms import QSVM
from qiskit.aqua import QuantumInstance
import numpy as np
import matplotlib.pyplot as plt


class QSVM_routine:

    def __init__(self,
                 feature_dim=2,
                 feature_depth=2,
                 train_test_split=0.3,
                 train_samples=5,
                 test_samples=2,
                 seed=0,
                 copies=5):
        self.feature_dim = feature_dim
        self.feature_depth = feature_depth
        self.train_test_split = train_test_split
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.seed = seed
        self.copies = copies

    # Create train test datasets

    def train_test_datasets(self):
        self.class_labels = [r'A', r'B']
        data, target = datasets.load_breast_cancer(True)
        train_X, test_X, train_y, test_y = train_test_split(data, target,
                                           test_size=self.train_test_split,
                                           random_state=self.seed)
        # Mean std normalization 
        self.z_scale = StandardScaler().fit(train_X)
        self.train_X_norm = self.z_scale.transform(train_X)
        self.test_X_norm = self.z_scale.transform(test_X)

        # Project the data into dimensions equal to the 
        # number of qubits
        self.pca = PCA(n_components=self.feature_dim).fit(self.train_X_norm)
        self.train_X_norm = self.pca.transform(self.train_X_norm)
        self.test_X_norm = self.pca.transform(self.test_X_norm)

        # Scale to the range (-1,+1)
        X_all = np.append(self.train_X_norm, self.test_X_norm, axis=0)
        minmax_scale = MinMaxScaler((-1, 1)).fit(X_all)
        self.train_X_norm = minmax_scale.transform(self.train_X_norm)
        self.test_X_norm = minmax_scale.transform(self.test_X_norm)

        # Pick training and test number of datapoint 
        self.train = {key: (self.train_X_norm[train_y == k, :])[:self.train_samples] for k, key in
                      enumerate(self.class_labels)}
        self.test = {key: (self.test_X_norm[test_y == k, :])[:self.test_samples] for k, key in
                     enumerate(self.class_labels)}

        
    # Train the QSVM Model
    def train_model(self):
        backend = Aer.get_backend('qasm_simulator')
        feature_expansion = SecondOrderExpansion(feature_dimension=self.feature_dim,
                                                 depth=self.feature_depth,
                                                 entangler_map=[[0, 1]])
        #feature_expansion = FirstOrderExpansion(feature_dimension=self.feature_dim)

        # Model definition
        svm = QSVM(feature_expansion, self.train, self.test)
        #svm.random_seed = self.seed
        q_inst = QuantumInstance(backend, shots=self.copies)

        # Train the SVM
        result = svm.run(q_inst)
        return svm, result

    # Analyze the training and test results

    def analyze_training_and_inference(self, result, svm):
        data_kernel_matrix = result['kernel_matrix_training']
        image = plt.imshow(np.asmatrix(data_kernel_matrix),
                           interpolation='nearest',
                           origin='upper',
                           cmap='bone_r')
        plt.show()
        print(f"Test Accuracy: {result['testing_accuracy']}")

    def main(self):
        self.train_test_datasets()
        svm, result = self.train_model()
        self.analyze_training_and_inference(svm, result)


if __name__ == '__main__':
    qsvm = QSVM_routine()
    qsvm.main()
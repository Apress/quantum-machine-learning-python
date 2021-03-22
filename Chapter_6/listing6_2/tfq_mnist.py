import tensorflow as tf
print('tensorflow version',tf.__version__)
import tensorflow_quantum as tfq
print('tensorflow quantum version',tfq.__version__)

import cirq
print('cirq version ',cirq.__version__)
import sympy
print('sympy version',sympy.__version__)
import numpy as np
print('numpy version',np.__version__)
import seaborn as sns
import collections

# visualization tools
import matplotlib.pyplot as plt
from cirq.contrib.svg import SVGCircuit


def extract_specific_digits(X, y, labels_to_extract):
    label_y1 = labels_to_extract[0]
    label_y2 = labels_to_extract[1]

    mask = (y == label_y1) | (y == label_y2)
    X, y = X[mask], y[mask]
    y = (y == label_y1)
    return X, y


def remove_sample_with_2_labels(X, y):
    mapping = collections.defaultdict(set)
    # Determine the set of labels for each unique image:
    for _x_, _y_ in zip(X, y):
        mapping[tuple(_x_.flatten())].add(_y_)

    new_x = []
    new_y = []
    for _x_, _y_ in zip(X, y):
        labels = mapping[tuple(_x_.flatten())]
        if len(labels) == 1:
            new_x.append(_x_)
            new_y.append(list(labels)[0])
        else:
            pass

    print("Initial number of examples: ", len(X))
    print("Final number of non-contradictory examples: ", len(new_x))

    return np.array(new_x), np.array(new_y)


def data_preprocessing(labels_to_extract, resize_dim=4, binary_threshold=0.5):
    # Load the data 
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    # Rescale the images from 0 to 1 range
    x_train = x_train[..., np.newaxis] / 255.0
    x_test = x_test[..., np.newaxis] / 255.0

    print("Number of original training examples:", len(x_train))
    print("Number of original test examples:", len(x_test))

    # Extract on the specified 2 classes  in labels_to_extract
    x_train, y_train = extract_specific_digits(x_train, y_train,
                                               labels_to_extract=labels_to_extract)
    x_test, y_test = extract_specific_digits(x_test, y_test,
                                             labels_to_extract=labels_to_extract)

    print("Number of filtered training examples:", len(x_train))
    print("Number of filtered test examples:", len(x_test))

    # Resize the MNIST Images since 28x28 size image requires as 
    # many qubits which is too much for Quantum Computers to 
    # allocate. We resize them to 4x4 for keeping the problem 
    # tractable in Quantum Computing realm.

    x_train_resize = tf.image.resize(x_train, (resize_dim, resize_dim)).numpy()
    x_test_resize = tf.image.resize(x_test, (resize_dim, resize_dim)).numpy()

    # Because of resizing to such small dimension there is a chance of images 
    # with different classes hashing to the same label. We remove such 
    # images below

    x_train_resize, y_train_resize = \
        remove_sample_with_2_labels(x_train_resize, y_train)

    # We represent each pixel in binary by applying a threshold
    x_train_bin = np.array(x_train_resize > binary_threshold, dtype=np.float32)
    x_test_bin = np.array(x_test_resize > binary_threshold, dtype=np.float32)

    return x_train_bin, x_test_bin, x_train_resize, x_test_resize, \
           y_train_resize, y_test


# Quantum circuit to represents each 0 valued pixel by |0> state 
# and 1 pixel by |1> state. 

def classical_to_quantum_data_circuit(image):
    image_flatten = image.flatten()
    qubits = cirq.GridQubit.rect(4, 4)
    circuit = cirq.Circuit()
    for i, val in enumerate(image_flatten):
        if val:
            circuit.append(cirq.X(qubits[i]))
    return circuit


# Define circuit for classical to quantum data  for all datapoints 
# and transfrom those cicuits to Tensors using Tensorflow Quantum

def classical_data_to_tfq_tensors(x_train_bin, x_test_bin):
    x_train_c = [classical_to_quantum_data_circuit(x) for x in x_train_bin]
    x_test_c = [classical_to_quantum_data_circuit(x) for x in x_test_bin]
    x_train_tfc = tfq.convert_to_tensor(x_train_c)
    x_test_tfc = tfq.convert_to_tensor(x_test_c)
    return x_train_tfc, x_test_tfc


class QuantumLayer:
    def __init__(self, data_qubits, readout):
        self.data_qubits = data_qubits
        self.readout = readout

    def add_layer(self, circuit, gate, prefix):
        for i, q in enumerate(self.data_qubits):
            _w_ = sympy.Symbol(prefix + '-' + str(i))
            circuit.append(gate(q, self.readout) ** _w_)


def create_QNN(resize_dim=4):
    """Create a QNN model circuit and prediction(readout) """
    data_qubits = cirq.GridQubit.rect(resize_dim, resize_dim)  # a 4x4 grid.
    readout = cirq.GridQubit(-1, -1)  # a single qubit at [-1,-1]
    circuit = cirq.Circuit()

    # Prepare the readout qubit.
    circuit.append(cirq.X(readout))
    circuit.append(cirq.H(readout))

    builder = QuantumLayer(
        data_qubits=data_qubits,
        readout=readout)

    # Apply a series of XX layers followed 
    # by a series of ZZ layers
    builder.add_layer(circuit, cirq.XX, "XX")
    builder.add_layer(circuit, cirq.ZZ, "ZZ")

    # Hadamard Gate on the readout qubit
    circuit.append(cirq.H(readout))

    return circuit, cirq.Z(readout)


def hinge_accuracy(y_true, y_pred):
    y_true = tf.squeeze(y_true) > 0.0
    y_pred = tf.squeeze(y_pred) > 0.0
    cost = tf.cast(y_true == y_pred, tf.float32)

    return tf.reduce_mean(cost)


def build_model(resize_dim=4):
    model_circuit, model_readout = create_QNN(resize_dim=resize_dim)
    # Build the model.
    model = tf.keras.Sequential([
        # The input is the data-circuit encoded as Tensors 
        tf.keras.layers.Input(shape=(), dtype=tf.string),
        # The PQC layer returns the expected value of the readout gate
        # in the range range [-1,1] 
        tfq.layers.PQC(model_circuit, model_readout),
    ])
    return model, model_circuit, model_readout


def main(labels_to_extract,
         resize_dim,
         binary_threshold,
         subsample,
         epochs=3,
         batch_size=32,
         eval=True):
    # Perform data preprocessing 
    x_train_bin, x_test_bin, x_train_resize, x_test_resize, \
    y_train_resize, y_test_resize = \
        data_preprocessing(labels_to_extract=labels_to_extract,
                           resize_dim=resize_dim,
                           binary_threshold=binary_threshold)

    x_train_tfc, x_test_tfc = \
        classical_data_to_tfq_tensors(x_test_bin, x_test_bin)

    # Convert labels to -1 or 1 to align with hinge loss
    y_train_hinge = 2.0 * y_train_resize - 1.0
    y_test_hinge = 2.0 * y_test_resize - 1.0

    # build model
    model, model_circuit, model_readout = \
        build_model(resize_dim=resize_dim)

    # Compile Model

    model.compile(
        loss=tf.keras.losses.Hinge(),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=[hinge_accuracy])
    print(model.summary())

    if subsample > 0:
        x_train_tfc_sub = x_train_tfc[:subsample]
        y_train_hinge_sub = y_train_hinge[:subsample]

    qnn_hist = model.fit(
        x_train_tfc_sub,
        y_train_hinge_sub,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_data=(x_test_tfc,
                         y_test_hinge))

    if eval:
        results = model.evaluate(x_test_tfc, y_test_hinge)
        print(results)


if __name__ == '__main__':
    labels_to_extract = [3, 6]
    resize_dim = 4
    binary_threshold = 0.5
    subsample = 500
    epochs = 3
    batch_size = 32

    main(labels_to_extract=labels_to_extract,
         resize_dim=resize_dim,
         binary_threshold=binary_threshold,
         subsample=subsample,
         epochs=epochs,
         batch_size=batch_size)
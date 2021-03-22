import cirq
print('cirq version',cirq.__version__)
import numpy as np
print('numpy version',np.__version__)
import scipy
from scipy.optimize import minimize
print('scipy version',scipy.__version__)


def setup_vqe(hamiltonian_bases=['ZZZ'], hamiltonian_scales=[-1.0]):

    num_qubits = len(hamiltonian_bases[0])
    eigen_values_dict = {}

    for base,scale in zip(hamiltonian_bases,hamiltonian_scales):
        eigen_values = []
        for i, char in enumerate(base):
            if char == 'Z':
                eigens = np.array([1, -1])
            elif char == 'I':
                eigens = np.array([1, 1])
            else:
                raise NotImplementedError(f"The Gate {char} is yet to be implemented")

            if len(eigen_values) == 0:
                eigen_values = eigens
            else:
                eigen_values = np.outer(eigen_values, eigens).flatten()

        eigen_values_dict_elem = {}

        for i, x in enumerate(list(eigen_values)):
            eigen_values_dict_elem[i] = scale * x

        eigen_values_dict[base] = eigen_values_dict_elem


    return eigen_values_dict, num_qubits


def ansatz_parameterized(theta,num_qubits=3):
    """
    Create an Ansatz
    :param theta: 
    :param num_qubits: 
    :return: 
    """
    qubits = [cirq.LineQubit(c) for c in range(num_qubits)]
    circuit = cirq.Circuit()
    for i in range(num_qubits):
        circuit.append(cirq.ry(theta[i]*np.pi)(qubits[i]))
    circuit.append(cirq.measure(*qubits, key='m'))
    print(circuit)
    return circuit, qubits


def compute_expectation(circuit, eigen_value_dict={}, copies=10000) -> float:
    sim = cirq.Simulator()
    results = sim.run(circuit, repetitions=copies)
    output = dict(results.histogram(key='m'))
    print('Stats', output)
    _expectation_ = 0
    for base in list(eigen_value_dict.keys()):
        for i in list(output.keys()):
            _expectation_ += eigen_value_dict[base][i] * output[i]

    _expectation_ = _expectation_ / copies

    return _expectation_


def VQE_routine(hamiltonian_bases=['ZZZ'], hamiltonian_scales=[1.], copies=1000,
         initial_theta=[0.5, 0.5, 0.5], verbose=True):
    eigen_value_dict, num_qubits = setup_vqe(hamiltonian_bases=hamiltonian_bases,
                                             hamiltonian_scales=hamiltonian_scales)
    print(eigen_value_dict)
    initial_theta = np.array(initial_theta)

    def objective(theta):
        circuit, qubits = ansatz_parameterized(theta, num_qubits)
        expectation = compute_expectation(circuit, eigen_value_dict, copies)
        if verbose:
            print(f" Theta: {theta} Expectation: {expectation}")
        return expectation

    result = minimize(objective, x0=initial_theta, method='COBYLA')
    print(result)
    circuit, _ = ansatz_parameterized(result.x, num_qubits)
    sim = cirq.Simulator()
    results = sim.run(circuit, repetitions=copies)
    stats = dict(results.histogram(key='m'))
    return result.x, result.fun, stats


if __name__ == '__main__':
    optim_theta, optim_func, hist_stats = VQE_routine(hamiltonian_bases=['IIZZ','IZIZ','IZZI','ZIIZ','ZZII'], hamiltonian_scales=[0.5,.5,.50,.50,.50],
                                          initial_theta=[0.5, 0.5,0.5,0.5])
    print(f"VQE Results: Minimum Hamiltonian Energy:{optim_func} at theta: {optim_theta}")
    print(f"Histogram for optimized State:", hist_stats)
    


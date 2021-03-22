import cirq
import numpy as np
print('cirq version',cirq.__version__)
print('numpy version',np.__version__)

def oracle(data_reg, y_reg, circuit, is_balanced=True):
    if is_balanced:
        circuit.append([cirq.CNOT(data_reg[0], y_reg), cirq.CNOT(data_reg[1], y_reg)])

    return circuit


def deutsch_jozsa(domain_size: int, func_type_to_simulate: str = "balanced", copies: int = 1000):
    """
    
    :param domain_size: Number of inputs to the function
    :param oracle: Oracle simulating the function
    :return: whether the function is balanced or constant
    """
    #  Define the data register and the target qubit

    reqd_num_qubits = int(np.ceil(np.log2(domain_size)))
    #Define the input qubits 
    data_reg = [cirq.LineQubit(c) for c in range(reqd_num_qubits)]
    # Define the Target qubits
    y_reg = cirq.LineQubit(reqd_num_qubits)
    # Define cirq Circuit
    circuit = cirq.Circuit()
    # Define equal superposition state for the input qubits
    circuit.append(cirq.H(data_reg[c]) for c in range(reqd_num_qubits))
    # Define Minus superposition state 
    circuit.append(cirq.X(y_reg))
    circuit.append(cirq.H(y_reg))
    
    # Check for nature of function : balanced/constant to simulate 
    # and implement Oracle accordingly 
    if func_type_to_simulate == 'balanced':
        is_balanced = True
    else:
        is_balanced = False
         
    circuit = oracle(data_reg, y_reg, circuit, is_balanced=is_balanced)
    # Apply Hadamard transform on each of the input qubits
    circuit.append(cirq.H(data_reg[c]) for c in range(reqd_num_qubits))
    # Measure the input qubits
    circuit.append(cirq.measure(*data_reg, key='z'))
    print("Circuit Diagram Follows")
    print(circuit)
    sim = cirq.Simulator()
    result = sim.run(circuit, repetitions=copies)
    print(result.histogram(key='z'))


if __name__ == '__main__':
    print("Execute Deutsch Jozsa for a Balanced Function of Domain size 4")
    deutsch_jozsa(domain_size=4, func_type_to_simulate='balanced', copies=1000)

    print("Execute Deutsch Jozsa for a Constant Function of Domain size 4")
    deutsch_jozsa(domain_size=4, func_type_to_simulate='', copies=1000)

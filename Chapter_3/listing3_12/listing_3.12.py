import cirq
print('cirq version',cirq.__version__)

def oracle(input_qubits, target_qubit, circuit, secret_element='01'):
    print(f"Element to be searched: {secret_element}")
    
    # Flip the qubits corresponding to the bits containing 0 
    for i, bit in enumerate(secret_element):
        if int(bit) == 0:
            circuit.append(cirq.X(input_qubits[i]))
    # Do a Conditional NOT using all input qubits as control
    circuit.append(cirq.TOFFOLI(*input_qubits, target_qubit))
    # Revert the input qubits to the state prior to Flipping 
    for i, bit in enumerate(secret_element):
        if int(bit) == 0:
            circuit.append(cirq.X(input_qubits[i]))
    return circuit


def grovers_algorithm(num_qubits=2, copies=1000):
    # Define input and Target Qubit
    input_qubits = [cirq.LineQubit(i) for i in range(num_qubits)]
    target_qubit = cirq.LineQubit(num_qubits)
    # Define Quantum Circuit
    circuit = cirq.Circuit()
    # Create equal Superposition State
    circuit.append([cirq.H(input_qubits[i]) for i in range(num_qubits)])
    # Take target qubit to minus state |->
    circuit.append([cirq.X(target_qubit), cirq.H(target_qubit)])
    # Pass the qubit through the Oracle
    circuit = oracle(input_qubits, target_qubit, circuit)
    # Construct Grover operator.
    circuit.append(cirq.H.on_each(*input_qubits))
    circuit.append(cirq.X.on_each(*input_qubits))
    circuit.append(cirq.H.on(input_qubits[1]))
    circuit.append(cirq.CNOT(input_qubits[0], input_qubits[1]))
    circuit.append(cirq.H.on(input_qubits[1]))
    circuit.append(cirq.X.on_each(*input_qubits))
    circuit.append(cirq.H.on_each(*input_qubits))

    # Measure the result.
    circuit.append(cirq.measure(*input_qubits, key='Z'))
    print("Grover's algorithm follows")
    print(circuit)
    sim = cirq.Simulator()
    result = sim.run(circuit, repetitions=copies)
    out = result.histogram(key='Z')
    
    out_result = {}
    for k in out.keys():
        new_key = "{0:b}".format(k)
        if len(new_key) < num_qubits:
            new_key = (num_qubits - len(new_key)) * '0' + new_key
        # print(new_key,k)
        out_result[new_key] = out[k]
    print(out_result)
    
    
    

if __name__ =='__main__':
    grovers_algorithm(2)
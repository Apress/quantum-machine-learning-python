import cirq
print('cirq version',cirq.__version__)

def oracle(input_qubits,target_qubits,circuit):
    # Oracle for Secret Code 110
    circuit.append(cirq.CNOT(input_qubits[2],target_qubits[1]))
    circuit.append(cirq.X(target_qubits[0]))
    circuit.append(cirq.CNOT(input_qubits[2], target_qubits[0]))
    circuit.append(cirq.CCNOT(input_qubits[0],input_qubits[1],target_qubits[0]))
    circuit.append(cirq.X(input_qubits[0]))
    circuit.append(cirq.X(input_qubits[1]))
    circuit.append(cirq.CCNOT(input_qubits[0], input_qubits[1], target_qubits[0]))
    circuit.append(cirq.X(input_qubits[0]))
    circuit.append(cirq.X(input_qubits[1]))
    circuit.append(cirq.X(target_qubits[0]))
    return circuit
    
    

def simons_algorithm_circuit(num_qubits=3):
    """
    Build the circuit for Simon's Algorithm
    :param num_qubits: 
    :return: 
    """
    input_qubits = [cirq.LineQubit(i) for i in range(num_qubits)]
    target_qubits = [cirq.LineQubit(k) for k in range(num_qubits, 2 * num_qubits)]
    circuit = cirq.Circuit()
    # Create Equal Superposition state for the Input Qubits
    circuit.append([cirq.H(input_qubits[i]) for i in range(num_qubits)])
    circuit = oracle(input_qubits, target_qubits, circuit)
    circuit.append(cirq.measure(*target_qubits, key='T'))
    circuit.append([cirq.H(input_qubits[i]) for i in range(num_qubits)])
    circuit.append(cirq.measure(*(input_qubits + target_qubits), key='Z'))
    print("Circuit Diagram for Simons Algorithm follows")
    print(circuit)
    #Simulate Algorithm 
    sim = cirq.Simulator()
    result = sim.run(circuit,repetitions=1000)
    out = dict(result.histogram(key='Z'))
    out_result = {}
    for k in out.keys():
        new_key =  "{0:b}".format(k)
        if len(new_key) < 2*num_qubits:
            new_key = (2*num_qubits - len(new_key))*'0' + new_key
        #print(new_key,k)
        out_result[new_key] = out[k]
    print(out_result)
    
    
if __name__ =='__main__':
    simons_algorithm_circuit()

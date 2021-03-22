import cirq
import numpy as np
print('cirq version',cirq.__version__)
print('numpy version',np.__version__)


def bell_inequality_test_circuit():
    """
    Define 4 qubits
    0th qubit - Alice
    1st qubit - contains the bit sent to Alice by the referee
    2nd qubit - Bob's qubit
    3rd qubit - contains the bit sent to Bob by the referee
        :return: 
    """
    qubits = [cirq.LineQubit(i) for i in range(4)]
    circuit = cirq.Circuit()
    # Entangle Alice and Bob to the Bell State
    circuit.append([cirq.H(qubits[0]), cirq.CNOT(qubits[0], qubits[2])])
    # Apply X^(-0.25) on Alice's Qubit
    circuit.append([cirq.X(qubits[0])**(-0.25)])
    # Apply Hadamard transform to the referee Qubits for Alice and Bob
    # This is done to randomize the qubit
    circuit.append([cirq.H(qubits[1]), cirq.H(qubits[3])])
    # Perform a Conditional X^0.5 on Alice and Bob Qubits based on corresponding referee qubits
    circuit.append([cirq.CNOT(qubits[1], qubits[0])**0.5])
    circuit.append([cirq.CNOT(qubits[3], qubits[2])**0.5])
    # Measure all the qubits
    circuit.append(cirq.measure(qubits[0], key='A'))
    circuit.append(cirq.measure(qubits[1], key='r_A'))
    circuit.append(cirq.measure(qubits[2], key='B'))
    circuit.append(cirq.measure(qubits[3], key='r_B'))
    return circuit

def main(iters=1000):
    # Build the Bell inequality test circuit
    circuit = bell_inequality_test_circuit()
    print("Bell Inequality Test Circuit")
    print(circuit)
    #Simulate for several iterations
    sim = cirq.Simulator()
    result = sim.run(circuit, repetitions=iters)
    A = result.measurements['A'][:, 0]
    r_A = result.measurements['r_A'][:, 0]
    B = result.measurements['B'][:, 0]
    r_B = result.measurements['r_B'][:, 0]
    
    win = (np.array(A) + np.array(B)) % 2 == (np.array(r_A) & np.array(r_B))
    print(f"Alice and Bob won {100*np.mean(win)} % of the times")

if __name__ == '__main__':
    main()
    
 
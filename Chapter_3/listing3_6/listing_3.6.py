import cirq
print('cirq version',cirq.__version__)


def quantum_teleportation(qubit_to_send_op='H'):
    q_register = [cirq.LineQubit(i) for i in range(3)]
    cirquit = cirq.Circuit()
    """
    Qubit 0 : Alice State qubit to be sent to Bob
    QUbit 1: Alices control qubit
    Qubit 2: Bobs control qubit 
    Set a state for Qubit 0 based on qubit_to_send_op : Implemented operators H,X,Y,Z,I
    """ 
    if qubit_to_send_op == 'H':
        cirquit.append(cirq.H(q_register[0]))
    elif qubit_to_send_op == 'X':
        cirquit.append(cirq.X(q_register[0]))
    elif qubit_to_send_op == 'Y':
        cirquit.append(cirq.X(q_register[0]))
    elif qubit_to_send_op == 'I':
        cirquit.append(cirq.I(q_register[0]))
    else:
        raise NotImplementedError("Yet to be implemented")
    
    # Entangle Alice and Bob's control qubits : Qubit 1 and Qubit 2
    cirquit.append(cirq.H(q_register[1]))
    cirquit.append(cirq.CNOT(q_register[1],q_register[2]))
    # CNOT Alice data qubit with control qubit
    cirquit.append(cirq.CNOT(q_register[0],q_register[1]))
    # Tranform Alices data qubit |+> |-> basis and perform measurement
    # on both of Alice's qubit
    cirquit.append(cirq.H(q_register[0]))
    cirquit.append(cirq.measure(q_register[0], q_register[1]))
    # Do a CNOT with Alice's control qubit on Bob's control qubit
    cirquit.append(cirq.CNOT(q_register[1],q_register[2]))
    # Do a Control Z on Bob's qubit based on Alice's data qubit
    cirquit.append(cirq.CZ(q_register[0],q_register[2]))
    cirquit.append(cirq.measure(q_register[2],key='Z'))
    print("Circuit")
    print(cirquit)
    sim = cirq.Simulator()
    output = sim.run(cirquit, repetitions=100)
    print("Measurement Output")
    print(output.histogram(key='Z'))
    
if __name__ == '__main__':
    quantum_teleportation(qubit_to_send_op='H')


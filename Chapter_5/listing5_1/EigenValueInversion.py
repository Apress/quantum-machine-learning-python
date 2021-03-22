import cirq
import numpy as np
import math

class EigenValueInversion(cirq.Gate):
    """
    Rotates the ancilla bit around the Y axis 
    by an angle theta = 2* sin inv(C/eigen value)
    corresponding to each Eigen value state basis |eigen value>. 
    This rotation brings the factor (1/eigen value) in
    the amplitude of the basis |1> of the ancilla qubit
    """

    def __init__(self, num_qubits, C, t):
        super(EigenValueInversion, self)
        self._num_qubits = num_qubits
        self.C = C
        self.t = t
        # No of possible Eigen values self.N
        self.N = 2**(num_qubits-1)

    def num_qubits(self):
        return self._num_qubits

    def _decompose_(self, qubits):
        """
        Apply the Rotation Gate for each possible 
        # Eigen value corresponding to the Eigen
        # value basis state. For each input basis state 
        # only the Rotation gate corresponding to it would be 
        # applied to the ancilla qubit
        """
        base_state = 2**self.N - 1

        for eig_val_state in range(self.N):
            eig_val_gate = self._ancilla_rotation(eig_val_state)

            if (eig_val_state != 0):
                base_state = eig_val_state - 1
            # XOR between successive eigen value states to 
            # determine the qubits  to flip
            qubits_to_flip = eig_val_state ^ base_state

            # Apply the flips to the qubits as determined 
            # by the XOR operation 

            for q in qubits[-2::-1]:

                if qubits_to_flip % 2 == 1:
                    yield cirq.X(q)
                qubits_to_flip >>= 1

                # Build controlled ancilla rotation
                eig_val_gate = cirq.ControlledGate(eig_val_gate)
            # Controlled Rotation Gate with the 1st (num_qubits -1) qubits as
            # control qubit and the last qubit as the target qubit(ancilla)

            yield eig_val_gate(*qubits)

    def _ancilla_rotation(self, eig_val_state):
        if eig_val_state == 0:
            eig_val_state = self.N
        theta = 2*math.asin(self.C * self.N * self.t / (2*np.pi * eig_val_state))
        # Rotation around the y axis by angle theta 
        return cirq.ry(theta)
    
def test(num_qubits=5):
    num_input_qubits = num_qubits - 1
    # Define ancila qubit
    ancilla_qubit = cirq.LineQubit(0)
    input_qubits = [cirq.LineQubit(i) for i in range(1, num_qubits)]
    #Define a circuit
    circuit = cirq.Circuit()
    # Set the state to equal superposition of |00000> and |00001>
    circuit.append(cirq.X(input_qubits[-4]))
    # t is set to 1 
    t = 0.358166*np.pi
    # Set C to the smallest Eigen value that can be measured
    C = 2 * np.pi / ((2 ** num_input_qubits) * t)
    circuit.append(EigenValueInversion(num_qubits,C,t)(*(input_qubits + [ancilla_qubit])))
    # Simulate circuit 
    sim = cirq.Simulator()
    result = sim.simulate(circuit)
    print(result)
    

if __name__ == '__main__':
    test(num_qubits=5)
    

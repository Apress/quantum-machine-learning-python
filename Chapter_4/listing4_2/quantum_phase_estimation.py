import cirq
print('cirq version', cirq.__version__)
from quantum_fourier_transform import QFT

class quantum_phase_estimation:
    
    def __init__(self, num_input_state_qubits=3,
                 num_ancillia_qubits=5,
                 unitary_transform=None,
                 U=None,
                 input_state=None):

        
        self.num_ancillia_qubits = num_ancillia_qubits
        
        self.output_qubits = [cirq.LineQubit(i) for i in range(self.num_ancillia_qubits)]
        
        self.input_circuit = cirq.Circuit()
        self.input_state =  input_state
        
        if self.input_state is not None:
            self.num_input_qubits = len(self.input_state)
        else:
            self.num_input_qubits = num_input_state_qubits
            
        self.input_qubits = [cirq.LineQubit(i) for i in
                             range(self.num_ancillia_qubits,
                                   self.num_ancillia_qubits + num_input_state_qubits)]

        if self.input_state is not None:
            for i, c in enumerate(self.input_state):
                if int(c) == 1:
                    self.input_circuit.append(cirq.X(self.input_qubits[i]))
                    
        self.unitary_transform = unitary_transform
        if self.unitary_transform is None:
            self.U = cirq.I
        elif self.unitary_transform == 'custom':
            self.U = U
        elif self.unitary_transform == 'Z':
            self.U = cirq.CZ
        elif self.unitary_transform == 'X':
            self.U = cirq.CX
        else:
            raise NotImplementedError(f"self.unitary transform not Implemented")

        self.circuit = cirq.Circuit()
        
        
    def phase_1_create_circuit_iter(self):
        
        for i in range(self.num_ancillia_qubits):
            self.circuit.append(cirq.H(self.output_qubits[i]))
            _pow_ = 2**(self.num_ancillia_qubits - 1 - i)
            #_pow_ = 2 ** (i)
            for k in range(self.num_input_qubits):
                print(self.U)
                self.circuit.append(self.U(self.output_qubits[i], self.input_qubits[k])**_pow_)
        
        
    def inv_qft(self):
        self._qft_ = QFT(qubits=self.output_qubits)
        self._qft_.qft_circuit()


    def simulate_circuit(self,circ):
        sim = cirq.Simulator()
        result = sim.simulate(circ)
        return result   
    
def main(num_input_state_qubits=1,
                 num_ancillia_qubits=2,
                 unitary_transform='Z',
                 U=None,input_state='1'):
    
    _QP_ = quantum_phase_estimation(num_ancillia_qubits=num_ancillia_qubits,
                                    num_input_state_qubits=num_input_state_qubits,
                                    unitary_transform=unitary_transform,
                                    input_state=input_state)
    _QP_.phase_1_create_circuit_iter()
    
    _QP_.inv_qft()
    
    circuit = _QP_.circuit  + _QP_._qft_.inv_circuit
    if len(_QP_.input_circuit) > 0:
        circuit = _QP_.input_circuit + circuit

    print(circuit)
    result = _QP_.simulate_circuit(circuit)
    print(result)

if __name__ == '__main__':
    main()

import cirq
import numpy as np
import fire
import elapsedtimer
from elapsedtimer import ElapsedTimer
print('cirq version',cirq.__version__)
print('numpy version',np.__version__)
print('fire',fire.__version__)
print('elapsedtimer',elapsedtimer.__version__)


class QFT:
    """
    Quantum Fourier Transform
    Builds the QFT circuit iteratively 
    """

    def __init__(self, signal_length=16,
                 basis_to_transform='',
                 validate_inverse_fourier=False,
                 qubits=None):
        
        self.signal_length = signal_length
        self.basis_to_transform = basis_to_transform
        
        if qubits is None:
            self.num_qubits = int(np.log2(signal_length))
            self.qubits = [cirq.LineQubit(i) for i in range(self.num_qubits)]
        else:
            self.qubits = qubits
            self.num_qubits = len(self.qubits)

        self.qubit_index = 0
        self.input_circuit = cirq.Circuit()

        self.validate_inverse_fourier = validate_inverse_fourier
        self.circuit = cirq.Circuit()
        # if self.validate_inverse_fourier:
        self.inv_circuit = cirq.Circuit()

        for k, q_s in enumerate(self.basis_to_transform):
            if int(q_s) == 1:
                # Change the qubit state from 0 to 1 
                self.input_circuit.append(cirq.X(self.qubits[k]))

    def qft_circuit_iter(self):

        if self.qubit_index > 0:
            # Apply the rotations on the prior qubits
            # conditioned on the current qubit
            for j in range(self.qubit_index):
                diff = self.qubit_index - j + 1
                rotation_to_apply = -2.0 / (2.0 ** diff)
                self.circuit.append(cirq.CZ(self.qubits[self.qubit_index],
                                            self.qubits[j]) ** rotation_to_apply)
        # Apply the Hadamard Transform 
        # on current qubit
        self.circuit.append(cirq.H(self.qubits[self.qubit_index]))
        # set up the processing for next qubit
        self.qubit_index += 1

    def qft_circuit(self):

        while self.qubit_index < self.num_qubits:
            self.qft_circuit_iter()
            # See the progression of the Circuit built
            print(f"Circuit after processing Qubit: {self.qubit_index - 1} ")
            print(self.circuit)
        # Swap the qubits to match qft definititon
        self.swap_qubits()
        print("Circuit after qubit state swap:")
        print(self.circuit)
        # Create the inverse Fourier Transform Circuit
        self.inv_circuit = cirq.inverse(self.circuit.copy())

    def swap_qubits(self):
        for i in range(self.num_qubits // 2):
            self.circuit.append(cirq.SWAP(self.qubits[i], self.qubits[self.num_qubits - i - 1]))

    def simulate_circuit(self):
        sim = cirq.Simulator()
        result = sim.simulate(self.circuit)
        return result


def main(signal_length=16,
         basis_to_transform='0000',
         validate_inverse_fourier=False):

    # Instantiate QFT Class
    _qft_ = QFT(signal_length=signal_length,
                basis_to_transform=basis_to_transform,
                validate_inverse_fourier=validate_inverse_fourier)


    # Build the QFT Circuit
    _qft_.qft_circuit()

    # Create the input Qubit State

    if len(_qft_.input_circuit) > 0:
        _qft_.circuit = _qft_.input_circuit + _qft_.circuit

    if _qft_.validate_inverse_fourier:
        _qft_.circuit += _qft_.inv_circuit

    print("Combined Circuit")
    print(_qft_.circuit)
    # Simulate the circuit

    output_state = _qft_.simulate_circuit()
    # Print the Results
    print(output_state)


if __name__ == '__main__':
    with ElapsedTimer('Execute Quantum Fourier Transform'):
        fire.Fire(main)

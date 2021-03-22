import cirq
from hamiltonian_simulator import HamiltonianSimulation
from QuantumPhaseEstimation import QuantumPhaseEstimation
from EigenValueInversion import EigenValueInversion
import numpy as np
import sympy


class HHL:

    def __init__(self, hamiltonian, initial_state=None, initial_state_transforms=None, qpe_register_size=4, C=None, t=1):
        """
        :param hamiltonian: Hamiltonian to Simulate
        :param C: hyper parameter to Eigen Value Inversion
        :param t: Time for which Hamiltonian is simulated
        :param initial_state: |b>
        """
        self.hamiltonian = hamiltonian
        self.initial_state = initial_state
        self.initial_state_transforms = initial_state_transforms
        self.qpe_register_size = qpe_register_size
        self.C = C
        self.t = t

        const = self.t/np.pi
        self.t = const*np.pi
        if self.C is None:
            self.C = 2*np.pi / (2**self.qpe_register_size * t)


    def build_hhl_circuit(self):
        self.circuit = cirq.Circuit()
        self.ancilla_qubit = cirq.LineQubit(0)
        self.qpe_register = [cirq.LineQubit(i) for i in range(1, self.qpe_register_size+1)]
        if self.initial_state is None:
            self.initial_state_size = int(np.log2(self.hamiltonian.shape[0]))
            if self.initial_state_size == 1:
                self.initial_state = [cirq.LineQubit(self.qpe_register_size + 1)]
            else:
                self.initial_state = [cirq.LineQubit(i) for i in range(self.qpe_register_size + 1,
                                               self.qpe_register_size + 1 + self.initial_state_size)]

        for op in list(self.initial_state_transforms):
            print(op)
            self.circuit.append(op(self.initial_state[0]))

        # Define Unitary Operator simulating the Hamiltonian
        self.U = HamiltonianSimulation(_H_=self.hamiltonian, t=self.t)
        # Perform Quantum Phase Estimation
        _qpe_ = QuantumPhaseEstimation(input_qubits=self.initial_state,
                                       output_qubits=self.qpe_register, U=self.U)
        _qpe_.circuit()
        print(dir(_qpe_))
        print('CIRCUIT',_qpe_.circuit)
        self.circuit += _qpe_.circuit
        # Perform EigenValue Inversion
        _eig_val_inv_ = EigenValueInversion(num_qubits=self.qpe_register_size + 1, C=self.C, t=self.t)
        self.circuit.append(_eig_val_inv_(*(self.qpe_register + [self.ancilla_qubit])))
        #Uncompute the qpe_register to |0..0> state
        print(self.circuit)
        #print(_qpe_.circuit**(-1))
        self.circuit.append(_qpe_.circuit**(-1))
        self.circuit.append(cirq.measure(self.ancilla_qubit,key='a'))
        self.circuit.append([
            cirq.PhasedXPowGate(
                exponent=sympy.Symbol('exponent'),
                phase_exponent=sympy.Symbol('phase_exponent'))(*self.initial_state),
            cirq.measure(*self.initial_state, key='m')
        ])

        #sim = cirq.Simulator()
        #results = sim.simulate(self.circuit)
        #print(results)

    def simulate(self):
        simulator = cirq.Simulator()

        # Cases for measuring X, Y, and Z (respectively) on the memory qubit.
        params = [{
            'exponent': 0.5,
            'phase_exponent': -0.5
        }, {
            'exponent': 0.5,
            'phase_exponent': 0
        }, {
            'exponent': 0,
            'phase_exponent': 0
        }]

        results = simulator.run_sweep(self.circuit, params, repetitions=5000)

        for label, result in zip(('X', 'Y', 'Z'), list(results)):
            # Only select cases where the ancilla is 1.
            # TODO: optimize using amplitude amplification algorithm.
            # Github issue: https://github.com/quantumlib/Cirq/issues/2216
            expectation = 1 - 2 * np.mean(
                result.measurements['m'][result.measurements['a'] == 1])
            print('{} = {}'.format(label, expectation))

    def main():
        """
        Simulates HHL with matrix input, and outputs Pauli observables of the
        resulting qubit state |x>.
        Expected observables are calculated from the expected solution |x>.
        """

        # Eigendecomposition:
        #   (4.537, [-0.971555, -0.0578339+0.229643j])
        #   (0.349, [-0.236813, 0.237270-0.942137j])
        # |b> = (0.64510-0.47848j, 0.35490-0.47848j)
        # |x> = (-0.0662724-0.214548j, 0.784392-0.578192j)
        A = np.array([[4.30213466 - 6.01593490e-08j,
                       0.23531802 + 9.34386156e-01j],
                      [0.23531882 - 9.34388383e-01j,
                       0.58386534 + 6.01593489e-08j]])
        t = 0.358166 * math.pi
        register_size = 4
        input_prep_gates = [cirq.rx(1.276359), cirq.rz(1.276359)]
        expected = (0.144130, 0.413217, -0.899154)

        # Set C to be the smallest eigenvalue that can be represented by the
        # circuit.
        C = 2 * math.pi / (2 ** register_size * t)

        # Simulate circuit
        print("Expected observable outputs:")
        print("X =", expected[0])
        print("Y =", expected[1])
        print("Z =", expected[2])
        print("Actual: ")
        simulate(hhl_circuit(A, C, t, register_size, *input_prep_gates))


if __name__ == '__main__':
    A = np.array([[4.30213466 - 6.01593490e-08j,
                   0.23531802 + 9.34386156e-01j],
                  [0.23531882 - 9.34388383e-01j,
                   0.58386534 + 6.01593489e-08j]])
    t = 0.358166 * np.pi
    C = None
    qpe_register_size = 4
    initial_state_transforms = [cirq.rx(1.276359), cirq.rz(1.276359)]
    _hhl_ = HHL(hamiltonian=A,initial_state_transforms=initial_state_transforms,qpe_register_size=4)
    _hhl_.build_hhl_circuit()
    _hhl_.simulate()


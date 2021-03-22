import cirq
from quantum_fourier_transform import QFT
import numpy as np
from fractions import Fraction
print('cirq version',cirq.__version__)
print('numpy version',np.__version__)

def euclid_gcd(a, b):
    if b == 0:
        return a
    else:
        return euclid_gcd(b, a % b)


"""
The Period Finding Class computes the Period of functions
of the form f(x) = a^x mod N using Quantum Phase Estimation
Alternately we can say the algorithm finds the period of 
the element a mod N
"""


class PeriodFinding:

    def __init__(self,
                 ancillia_precision_bits=4,
                 func_domain_size=16,
                 a=7,
                 N=15
                 ):

        self.ancillia_precision_bits = ancillia_precision_bits
        self.func_domain_size = func_domain_size
        self.num_output_qubits = self.ancillia_precision_bits
        self.num_input_qubits = int(np.log2(self.func_domain_size))
        self.output_qubits = [cirq.LineQubit(i) for i in range(self.num_output_qubits)]
        self.input_qubits = [cirq.LineQubit(i) for i in range(self.num_output_qubits,
                                                              self.num_output_qubits + self.num_input_qubits)]
        self.a = a

        self.N = N
        if self.N is None:
            self.N = func_domain_size - 1

        self.circuit = cirq.Circuit()

    def periodic_oracle(self, a, m, k):
        """
        Implement an oracle U_a that takes in the state  
        input state |y> and outputs |ay mod N>
        """

        for i in range(m):
            if a in [2, 13]:
                self.circuit.append(cirq.SWAP(self.input_qubits[0],
                                              self.input_qubits[1]).controlled_by(self.output_qubits[k]))
                self.circuit.append(cirq.SWAP(self.input_qubits[1],
                                              self.input_qubits[2]).controlled_by(self.output_qubits[k]))
                self.circuit.append(cirq.SWAP(self.input_qubits[2],
                                              self.input_qubits[3]).controlled_by(self.output_qubits[k]))

            if a in [7, 8]:
                self.circuit.append(cirq.SWAP(self.input_qubits[2],
                                              self.input_qubits[3]).controlled_by(self.output_qubits[k]))
                self.circuit.append(cirq.SWAP(self.input_qubits[1],
                                              self.input_qubits[2]).controlled_by(self.output_qubits[k]))
                self.circuit.append(cirq.SWAP(self.input_qubits[0],
                                              self.input_qubits[1]).controlled_by(self.output_qubits[k]))
            if a in [4, 11]:
                self.circuit.append(cirq.SWAP(self.input_qubits[1],
                                              self.input_qubits[3]).controlled_by(self.output_qubits[k]))
                self.circuit.append(cirq.SWAP(self.input_qubits[0],
                                              self.input_qubits[2]).controlled_by(self.output_qubits[k]))

            if a in [7, 11, 13]:
                for j in range(self.num_input_qubits):
                    self.circuit.append(cirq.X(self.input_qubits[j]).controlled_by(self.output_qubits[k]))

    def build_phase_1_period_finding_circuit(self):

        # Apply Hadamard Transform on each output qubit

        self.circuit.append([cirq.H(self.output_qubits[i]) for i in range(self.num_output_qubits)])
        # Set input qubit to state |0001>
        self.circuit.append(cirq.X(self.input_qubits[-1]))

        if euclid_gcd(self.N, self.a) != 1:
            print(f"{self.a} is not co-prime to {self.N}")
            co_primes = []
            for elem in range(2, self.N):
                if euclid_gcd(self.N, elem) == 1:
                    co_primes.append(elem)
            print(f"Select a from the list of co-primes to {self.N}: {co_primes} ")

        else:
            print(f"Trying period finding of element a = {self.a} mod {self.N}")
            a = self.a

        for q in range(self.num_output_qubits):
            _pow_ = 2 ** (self.num_output_qubits - q - 1)
            self.periodic_oracle(a=a, m=_pow_, k=q)

    def inv_qft(self):
        """
        Inverse Fourier Transform
        :return: 
        IFT circuit
        """
        self._qft_ = QFT(qubits=self.output_qubits)
        self._qft_.qft_circuit()

    def simulate_circuit(self, circ):
        """
        Simulates the Period Finding Algorithm 
        :param circ: Circuit to Simulate 
        :return: Output results of Simulation 
        """
        circ.append([cirq.measure(*self.output_qubits, key='Z')])
        sim = cirq.Simulator()
        result = sim.run(circ, repetitions=1000)
        out = dict(result.histogram(key='Z'))
        out_result = {}
        for k in out.keys():
            new_key = "{0:b}".format(k)
            if len(new_key) < self.num_output_qubits:
                new_key = (self.num_output_qubits - len(new_key)) * '0' + new_key
            # print(new_key,k)
            out_result[new_key] = out[k]

        return out_result

    def measurement_to_period(self, results, denom_lim=15):
        # Convert a state to Phase as a binary fraction 
        # |x_1,x_2....x_n> -> x_1*2^-1 + x_2*2^-2 + .. + x_n*2^-n  
        measured_states = list(results.keys())

        measured_phase = []
        measured_phase_rational = []

        for s in measured_states:
            phase = int(s, 2) / (2 ** len(s))
            phase_rational = Fraction(phase).limit_denominator(denom_lim)
            measured_phase.append(phase)
            measured_phase_rational.append(phase_rational)
        print(f"---------------------------------")
        print(f"Measured  |   Real   |   Rational")
        print(f"State     |   Phase  |    Phase  ")
        print(f"---------------------------------")
        for i in range(len(measured_phase)):
            print(f"    {measured_states[i]}  |  {measured_phase[i]}    |  {measured_phase_rational[i]}")
            print(f"---------------------------------")
        print('\n')

        max_phase_rational = measured_phase_rational[np.argmax(np.array(measured_phase))]
        max_phase_numerator = max_phase_rational.numerator
        max_phase_denominator = max_phase_rational.denominator
        if (max_phase_denominator - max_phase_numerator) == 1:
            period = max_phase_denominator
        else:
            print(f"period cannot be determined")
            period = np.inf

        return period


def period_finding_routine(func_domain_size=16,
         ancillia_precision_bits=4,
         a=7,
         N=15):
    """

    :param func_domain_size:
        States in the Domain of the function.
    :param ancillia_precision_bits:
        Precision bits for Phase Measurement
    :param N: Generally func_domain_size - 1 
    :param a:  Element whose periodicity mod N 
               is to be computed
    :return: Period r of the element a mod N 
     """

    _PF_ = PeriodFinding(ancillia_precision_bits=ancillia_precision_bits,
                         func_domain_size=func_domain_size,
                         a=a,
                         N=N)
    _PF_.build_phase_1_period_finding_circuit()

    _PF_.inv_qft()

    circuit = _PF_.circuit + _PF_._qft_.inv_circuit

    print(circuit)
    result = _PF_.simulate_circuit(circuit)
    print(f"Measurement Histogram Results follow")
    print(result)
    print('\n')
    period = _PF_.measurement_to_period(result, denom_lim=_PF_.N)
    print(f"Period of {a} mod {N} is: {period} ")
    return period


if __name__ == '__main__':
    period_finding_routine()

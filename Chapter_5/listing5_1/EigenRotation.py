import cirq
import math



class EigenValueRotation(cirq.Gate):
    """
    EigenValueRotation rotates the an
    """

    def __init__(self, num_qubits, C, t):
        super(EigenRotation, self)
        self._num_qubits = num_qubits
        self.C = C
        self.t = t
        self.N = 2**(num_qubits-1)

    def num_qubits(self):
        return self._num_qubits

    def _decompose_(self, qubits):
        for k in range(self.N):
            if k == 0:
                k_base = 2**self.N - 1
            else:
                k_base = k - 1 
            kGate = self._ancilla_rotation(k)
           # xor's 1 bits correspond to X gate positions.
            xor = k ^ k_base
            Gate_apply = []  
            for q in qubits[-2::-1]:
                # Place X gates
                if xor % 2 == 1:
                    yield cirq.X(q)
                    Gate_apply.append('X')
                else:
                    Gate_apply.append(' ')
                xor >>= 1

                # Build controlled ancilla rotation
                kGate = cirq.ControlledGate(kGate)
            
            yield kGate(*qubits)
            print(Gate_apply)

    def _ancilla_rotation(self, k):
        if k == 0:
            k = self.N
        theta = 2*math.asin(self.C * self.N * self.t / (2*math.pi * k))
        return cirq.ry(theta)
def main(register_size=4):
    ancilla = cirq.LineQubit(0)
    register = [cirq.LineQubit(i + 1) for i in range(register_size)]
    print(register)
    memory = cirq.LineQubit(register_size + 1)
    t = 0.358166*math.pi
    C = 2 * math.pi / (2 ** register_size * t)
    print(math.asin(C*t))
    circuit = cirq.Circuit()
    #circuit.append(cirq.H(register[i]) for i in range(register_size))
    circuit.append(cirq.X(register[-3]))
    circuit.append(EigenRotation(register_size + 1, C, t)(*(register + [ancilla])))
    #circuit.append(cirq.measure(*(register + [ancilla]),key='m'))
    sim = cirq.Simulator()
    result = sim.simulate(circuit)
    #result = sim.run(circuit,repetitions=1000)
    print(result)
    print(circuit)
    #print(result.histogram(key='m'))
    
    
    
if __name__ == '__main__':
    main()
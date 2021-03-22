import cirq
import numpy as np
print('cirq version',cirq.__version__)
print('numpy version',np.__version__)


class SwapTest:
    def __init__(self,prepare_input_states=None,input_state_dim=None,nq=0,measure=False,copies=1000):
        self.nq = nq
        self.prepare_input_states = prepare_input_states
        self.input_state_dim = input_state_dim
        if input_state_dim is not None:
            self.num_qubits_input_states = int(np.log2(self.input_state_dim))
            print(self.num_qubits_input_states)
        
        
        self.measure = measure
        self.copies = copies
        self.ancilla_qubit = cirq.LineQubit(self.nq)
        self.nq += 1
        
        
        if self.prepare_input_states is not None:
            if input_state_dim is None:
                raise ValueError("Please enter a valid dimension for input states to compare")
            else:
                self.num_qubits_input_states = int(np.log2(self.input_state_dim))
                self.input_1 = [cirq.LineQubit(i) for i in range(self.nq, self.nq +self.num_qubits_input_states)]
                self.nq += self.num_qubits_input_states
                self.input_2 = [cirq.LineQubit(i) for i in range(self.nq,
                                                                 self.nq + self.num_qubits_input_states)]
                self.nq += self.num_qubits_input_states
    
    def build_circuit(self,input_1=None,input_2=None,input_1_transforms=None,input_2_transforms=None):
        
        self.circuit = cirq.Circuit()
        if input_1 is not None:
            self.input_1 = input_1
        if input_2 is not None:
            self.input_2 = input_2
        if input_1_transforms is not None:
            for op in input_1_transforms:
                print(op)
                print(self.input_1)
                self.circuit.append(op.on_each(self.input_1))
        if input_2_transforms is not None:
            for op in input_2_transforms:
                self.circuit.append(op.on_each(self.input_2))
        

        # Ancilla in + state         
        self.circuit.append(cirq.H(self.ancilla_qubit))
        # Swap states conditoned on the ancilla
        for i in range(len(self.input_1)):
            self.circuit.append(cirq.CSWAP(self.ancilla_qubit, self.input_1[i], 
                                           self.input_2[i]))
        # Hadamard Transform on Ancilla 
        self.circuit.append(cirq.H(self.ancilla_qubit))
        if self.measure:
            self.circuit.append(cirq.measure(self.ancilla_qubit,key='m'))
        print(self.circuit)
        
    def simulate(self):
        sim = cirq.Simulator()
        results = sim.run(self.circuit,repetitions=self.copies)
        results = results.histogram(key='m')
        prob_0 = results[0]/self.copies
        dot_product_sq = 2*(max(prob_0 - .5,0))
        return prob_0,dot_product_sq

def main(prepare_input_states=True,input_state_dim=4,
                  input_1_transforms=[cirq.H],
                  input_2_transforms=[cirq.I],measure=True,copies=1000):
    st = SwapTest(prepare_input_states=prepare_input_states,input_state_dim=input_state_dim,measure=measure,copies=copies)
    st.build_circuit(input_1_transforms=input_1_transforms,
                     input_2_transforms=input_2_transforms)
    prob_0, dot_product_sq = st.simulate()
    print(f"Probability of zero state {prob_0}")
    print(f"Sq of Dot product  {dot_product_sq}")
    print(f"Dot product  {dot_product_sq**0.5}")

if __name__ == '__main__':
    main()
    
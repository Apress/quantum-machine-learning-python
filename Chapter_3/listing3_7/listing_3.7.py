import cirq
import numpy as np
print('cirq verison',cirq.__version__)
print('numpy version',np.__version__)

def random_number_generator(low=0,high=2**10,m=20):
    """
    
    :param low: lower bound of numbers to be generated  
    :param high: Upper bound of numbers to be generated
    :param number m : Number of random numbers to output
    :return: string random numbers 
    """
    # Determine the number of Qubits required
    qubits_required = int(np.ceil(np.log2(high - low)))
    print(qubits_required)
    # Define the qubits
    Q_reg = [cirq.LineQubit(c) for c in range(qubits_required)]
    # Define the circuit
    circuit = cirq.Circuit()
    circuit.append(cirq.H(Q_reg[c]) for c in range(qubits_required))
    circuit.append(cirq.measure(*Q_reg, key="z"))
    print(circuit)
    
    # Simulate the cirquit 
    
    sim = cirq.Simulator()
    
    num_gen = 0
    output = []
    while num_gen <= m :
        result = sim.run(circuit,repetitions=1)
        rand_number = result.data.get_values()[0][0] + low
        if rand_number < high :
            output.append(rand_number)
            num_gen += 1
    return output


if __name__ == '__main__':
    output = random_number_generator()
    print("Sampled Random Numbers")
    print(output)
    print("Mean of the Sampled Random Numbers", np.mean(output))
    
        
        
    
    
    
    
    
    
    
    
    
    
 
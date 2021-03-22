# Import the cirq package  

import cirq
print('cirq version',cirq.__version__)
# Define a Qubit
qubit = cirq.GridQubit(0,0)

# Create a Cirquit
circuit = cirq.Circuit([cirq.H(qubit),
                                 cirq.measure(qubit,key='m')])
print("Circuit Follows")
print(circuit)

sim = cirq.Simulator()
output = sim.run(circuit,repetitions=100) 
print("Measurement Output:")
print(output)
print(output.histogram(key='m'))
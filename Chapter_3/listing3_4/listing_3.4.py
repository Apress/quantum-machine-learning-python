# Bell State Creation
import cirq
print('cirq version', cirq.__version__)
#Define the two qubits using LineQubit
q_register = [cirq.LineQubit(i) for i in range(2)]
cirquit = cirq.Circuit([cirq.H(q_register[0]), cirq.CNOT(q_register[0], q_register[1])])
cirquit.append(cirq.measure(*q_register,key='z'))
print("Circuit")
print(cirquit)
sim = cirq.Simulator()
output = sim.run(cirquit, repetitions=100)
print("Measurement Output")
print(output.histogram(key='z'))
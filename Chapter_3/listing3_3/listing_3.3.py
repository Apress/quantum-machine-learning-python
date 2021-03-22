"""
Measure a qubit after Hadamard Transform 
"""
import qiskit
print('qiskit version', qiskit.__version__)
from qiskit import QuantumCircuit, execute, Aer
from qiskit.visualization import plot_histogram

# Use Aer's qasm_simulator
simulator = Aer.get_backend('qasm_simulator')

# Create a Quantum Circuit with 1 Qubit 
circuit = QuantumCircuit(1, 1)

# Add a H gate on Qubit 0
circuit.h(0)

# Map the quantum measurement to the classical register 
circuit.measure([0], [0])

# Execute the circuit on the qasm simulator
job = execute(circuit, simulator, shots=100)

# Grab results from the job
result = job.result()

# Returns counts
counts = result.get_counts(circuit)
print("\nTotal count for 0 and 1 are:",counts)

# Draw the circuit
print(circuit.draw(output='text'))
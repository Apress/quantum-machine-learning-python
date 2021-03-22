import cirq
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
print('cirq version', cirq.__version__)
print('numpy version', np.__version__)
print('matplotlib version', matplotlib.__version__)
print('networkx version', nx.__version__)


class GraphQuantumRandomWalk:

    def __init__(self, graph_hamiltonian, t, verbose=True):
        self.graph_ham = graph_hamiltonian
        self.num_vertices = self.graph_ham.shape[0]
        self.num_qubits = int(np.log2(self.num_vertices))
        self.qubits = [cirq.LineQubit(i) for i in range(self.num_qubits)]
        self.t = t
        self.verbose = verbose

    @staticmethod
    def diagonal_exponential(qubits, eigen_vals, t):
        circuit = cirq.Circuit()
        q1 = qubits[0]
        q2 = qubits[1]
        circuit.append(cirq.CZ(q1, q2) ** (-eigen_vals[-1] * t / np.pi))
        circuit.append([cirq.X(q2), cirq.CZ(q1, q2) ** (-eigen_vals[-2] * t / np.pi), cirq.X(q2)])
        circuit.append([cirq.X(q1), cirq.CZ(q1, q2) ** (-eigen_vals[-3] * t / np.pi), cirq.X(q1)])
        circuit.append(
            [cirq.X(q1), cirq.X(q2), cirq.CZ(q1, q2) ** (-eigen_vals[-4] * t / np.pi), cirq.X(q1), cirq.X(q2)])
        return circuit

    def unitary(self):
        eigen_vals, eigen_vecs = np.linalg.eigh(self.graph_ham)
        idx = eigen_vals.argsort()[::-1]
        eigen_vals = eigen_vals[idx]
        eigen_vecs = eigen_vecs[:, idx]
        if self.verbose:
            print(f"The Eigen values: {eigen_vals}")

        self.circuit = cirq.Circuit()
        self.circuit.append(cirq.H.on_each(self.qubits))
        self.circuit += self.diagonal_exponential(self.qubits, eigen_vals, self.t)
        self.circuit.append(cirq.H.on_each(self.qubits))

    def simulate(self):
        sim = cirq.Simulator()
        results = sim.simulate(self.circuit).final_state
        prob_dist = [np.abs(a) ** 2 for a in results]
        return prob_dist

    def main(self):
        self.unitary()
        prob_dist = self.simulate()
        if self.verbose:
            print(f"The converged prob_dist: {prob_dist}")
        return prob_dist


if __name__ == '__main__':
    graph_hamiltonian = np.ones((4, 4))
    #graph_hamiltonian = np.array([[1, 1, 1, 0], [1, 1, 0, 1], [1, 0, 1, 1], [0, 1, 1, 1]])
    time_to_simulate = 4
    steps = 80
    time_trace = []
    prob_dist_trace = []
    for t in np.linspace(0, time_to_simulate):
        gqrq = GraphQuantumRandomWalk(graph_hamiltonian=graph_hamiltonian, t=t)
        prob_dist = gqrq.main()
        time_trace.append(t)
        prob_dist_trace.append(prob_dist)
    prob_dist_trace = np.array(prob_dist_trace)
    plt.plot(time_trace, prob_dist_trace[:, 0])
    plt.show()
    rows, cols = np.where(graph_hamiltonian == 1)
    edges = zip(rows.tolist(), cols.tolist())
    gr = nx.Graph()
    gr.add_edges_from(edges)
    nx.draw(gr,node_size=4)
    plt.show()
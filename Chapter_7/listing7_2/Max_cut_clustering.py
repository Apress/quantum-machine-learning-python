import cirq
from vqe_cirq import *
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


class QuantumMaxCutClustering:

    def __init__(self, adjacency_matrix: np.ndarray, invert_adjacency=True):
        self.adjacency_matrix = adjacency_matrix
        self.num_vertices = self.adjacency_matrix.shape[0]
        self.hamiltonian_basis_template = 'I' * self.num_vertices
        if invert_adjacency:
            self.hamiltonian = 1 - self.adjacency_matrix
        else:
            self.hamiltonian = self.adjacency_matrix

    def create_max_cut_hamiltonian(self):

        hamiltonian_bases, hamiltonian_coefficients = [], []
        for i in range(self.num_vertices):
            for j in range(i + 1, self.num_vertices):
                if self.hamiltonian[i, j] > 0:
                    hamiltonian_coefficients.append(self.hamiltonian[i, j])

                    hamiltonian_base = ''
                    for k, c in enumerate(self.hamiltonian_basis_template):
                        if k in [i, j]:
                            hamiltonian_base += 'Z'
                        else:
                            hamiltonian_base += self.hamiltonian_basis_template[k]
                    hamiltonian_bases.append(hamiltonian_base)
        return hamiltonian_bases, hamiltonian_coefficients

    def vqe_simulation(self, hamiltonian_bases,
                       hamiltonian_coefficients,
                       initial_theta=None,
                       copies=10000):
        if initial_theta is None:
            initial_theta = [0.5] * self.num_vertices
        optim_theta, optim_func, hist_stats = \
            VQE_routine(hamiltonian_bases=hamiltonian_bases,
                        hamiltonian_scales=hamiltonian_coefficients,
                        initial_theta=initial_theta,
                        copies=copies)
        solution_stat = max(hist_stats, key=hist_stats.get)
        solution_stat = bin(solution_stat).replace("0b", "")
        solution_stat = (self.num_vertices - len(solution_stat)) * "0" + solution_stat

        return optim_theta, optim_func, hist_stats, solution_stat

    def max_cut_cluster(self, distance_matrix, solution_state):
        print(distance_matrix)
        G = nx.Graph()
        G.add_nodes_from(np.arange(0, self.num_vertices, 1))
        edge_list = []
        for i in range(self.num_vertices):
            for j in range(i + 1, self.num_vertices):
                if distance_matrix[i, j] > 0:
                    edge_list.append((i, j, 1.0))
        G.add_weighted_edges_from(edge_list)
        colors = []
        for s in solution_state:
            if int(s) == 1:
                colors.append('r')
            else:
                colors.append('b')
        pos = nx.spring_layout(G)
        default_axes = plt.axes(frameon=True)
        nx.draw_networkx(G, node_color=colors, node_size=600, alpha=.8, ax=default_axes, pos=pos)
        plt.savefig('Maxcut_clustering.png')
        

    def main(self):
        hamiltonian_bases, hamiltonian_coefficients = self.create_max_cut_hamiltonian()
        print(hamiltonian_bases)
        optim_theta, optim_func, \
        hist_stats, solution_state = self.vqe_simulation(hamiltonian_bases,
                                                         hamiltonian_coefficients)

        print(f"VQE Results: Minimum Hamiltonian Energy:{optim_func} at theta: {optim_theta}")
        print(f"Histogram for optimized State:", hist_stats)
        print(f"Solution state: {solution_state}")
        self.max_cut_cluster(distance_matrix=self.hamiltonian, solution_state=solution_state)


if __name__ == '__main__':
    adjacency_matrix = np.array([[0, 0, 0, 0],
                                 [0, 0, 0, 1],
                                 [0, 0, 0, 0],
                                 [0, 1, 0, 0]])
    mc = QuantumMaxCutClustering(adjacency_matrix=adjacency_matrix)
    mc.main()

import cirq
import numpy as np


class HamiltonianSimulation(cirq.EigenGate, cirq.SingleQubitGate):
    """
    This class simulates the Hamiltonian evolution for
    a Single qubit. For a Hamiltonian given by H the Unitary Operator
    simulated for time t is given by e**(-iHt). The Eigen vectors of the 
    Hamiltonian H and  the Unitary operator. An eigen value of lambda for 
    the Hamiltonian H corresponds to the Eigen value of e**(-i*lambda*t)

    The EigenGate takes in an Eigenvalue of the form e**(i*pi*theta) as theta
    abd the corresponding Eigen vector as |v><v|
    """

    def __init__(self, _H_, t, exponent=1.0):
        cirq.SingleQubitGate.__init__(self)
        cirq.EigenGate.__init__(self, exponent=exponent)
        self._H_ = _H_
        self.t = t
        eigen_vals, eigen_vecs = np.linalg.eigh(self._H_)
        self.eigen_components = []
        for _lambda_, vec in zip(eigen_vals, eigen_vecs.T):
            theta = -_lambda_*t / np.pi
            _proj_ = np.outer(vec, np.conj(vec))
            self.eigen_components.append((theta, _proj_))

    def _with_exponent(self, exponent):
        return HamiltonianSimulation(self._H_, self.t, exponent)

    def _eigen_components(self):
        return self.eigen_components


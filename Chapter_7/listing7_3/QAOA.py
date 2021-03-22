import cirq
import numpy as np
print('cirq version',cirq.__version__)
print('np version', np.__version__)




class QAOA:

    def __init__(self, num_elems:int,
                 hamiltonian_type:str,
                 hamiltonian_interactions:np.ndarray,
                 verbose=True):
        self.num_elems = num_elems
        self.hamiltonian_type = hamiltonian_type
        self.hamiltonian_interactions = hamiltonian_interactions
        self.verbose = verbose
        if self.hamiltonian_type not in ['isling']:
            raise ValueError(f"No support for the Hamiltonian type {self.hamiltonian_type}")
        self.qubits = [cirq.LineQubit(x) for x in range(num_elems)]


    @staticmethod
    def interaction_gate(q1, q2, gamma=1):
        circuit = cirq.Circuit()
        circuit.append(cirq.CZ(q1, q2)**gamma)
        circuit.append([cirq.X(q2), cirq.CZ(q1, q2)**(-gamma), cirq.X(q2)])
        circuit.append([cirq.X(q1), cirq.CZ(q1, q2) **(-gamma), cirq.X(q1)])
        circuit.append([cirq.X(q1), cirq.X(q2), cirq.CZ(q1, q2) ** gamma, cirq.X(q1), cirq.X(q2)])
        return circuit

# Build the Target Hamiltonian based circuit Evolution
    def target_hamiltonian_evolution_circuit(self, gamma):
        circuit = cirq.Circuit()
        # Apply the interaction gates to all the qubit pairs

        for i in range(self.num_elems):

            for j in range(i+1, self.num_elems):
                circuit.append(self.interaction_gate(
                                    self.qubits[i], self.qubits[j],
                                    gamma=gamma))
        return circuit

# Build the Starting Hamiltonian based evolution circuit
    def starting_hamiltonian_evolution_circuit(self, beta):
        for i in range(self.num_elems):
            yield cirq.X(self.qubits[i])**beta

    def build_qoaa_circuit(self, gamma_store, beta_store):
        self.circuit = cirq.Circuit()
        # Hadamard gate on each qubit to get an equal superposition state
        print(self.qubits)
        self.circuit.append(cirq.H.on_each(self.qubits))

        for i in range(len(gamma_store)):
            self.circuit.append(self.target_hamiltonian_evolution_circuit(gamma_store[i]))
            self.circuit.append(self.starting_hamiltonian_evolution_circuit(beta_store[i]))

    def simulate(self):
        #print(self.circuit)
        sim = cirq.Simulator()
        waveform = sim.simulate(self.circuit)
        return waveform


    def expectation(self,waveform):

        expectation = 0
        prob_from_waveform = (np.absolute(waveform.final_state))**2
        #print(prob_from_waveform)
        for i in range(len(prob_from_waveform)):
            base = bin(i).replace("0b", "")
            base = (self.num_elems - len(base))*'0' + base
            base_array = []
            for b in base:
                if int(b) == 0:
                    base_array.append(-1)
                else:
                    base_array.append(1)

            base_array = np.array(base_array)
            base_interactions = np.outer(base_array, base_array)
            #print(i, prob_from_waveform[i], np.sum(np.multiply(base_interactions,self.hamiltonian_interactions)))
            expectation =+ prob_from_waveform[i]*np.sum(np.multiply(base_interactions,self.hamiltonian_interactions))
        return expectation

    def optimize_params(self, gammas, betas, verbose=True):
        expectation_dict = {}
        waveforms_dict = {}
        for i, gamma in enumerate(gammas):
            for j, beta in enumerate(betas):
                self.build_qoaa_circuit([gamma],[beta])
                waveform = self.simulate()
                expectation = self.expectation(waveform)
                expectation_dict[(gamma,beta)] = expectation
                waveforms_dict[(gamma,beta)] = waveform.final_state
                if verbose:
                    print(f"Expectation for gamma:{gamma}, beta:{beta} = {expectation}")
        return expectation_dict, waveforms_dict


    def main(self):
        gammas = np.linspace(0, 1,50)
        betas = np.linspace(0, np.pi, 50)
        expectation_dict, waveform_dict = self.optimize_params(gammas, betas)
        expectation_vals = np.array(list(expectation_dict.values()))
        expectation_params = list(expectation_dict.keys())
        waveform_vals = np.array(list(waveform_dict.values()))
        optim_param = expectation_params[np.argmin(expectation_vals)]
        optim_expectation = expectation_vals[np.argmin(expectation_vals)]
        optim_waveform = waveform_vals[np.argmin(expectation_vals)]
        print(f"Optimized parameters")
        print(f"-----------------------------")
        print(f"  gamma,beta = {optim_param[0]}, {optim_param[1]}")
        print(f"  Expectation = {optim_expectation}")
        print(f"-----------------------------")
        optimal_waveform_prob = np.array([np.abs(x)**2 for x in optim_waveform])
        states = np.array([bin(i).replace('0b', "") for i in range(2 ** self.num_elems)])
        states = np.array([((self.num_elems - len(s)) * '0' + s) for s in states])
        sort_indices = np.argsort(-1 * np.array(optimal_waveform_prob))
        print(sort_indices)
        optimal_waveform_prob = optimal_waveform_prob[sort_indices]
        states = states[sort_indices]
        print(f"State | Probability")
        print(f"-----------------------------")
        for i in range(len(states)):
            print(f"{states[i]} | {np.round(optimal_waveform_prob[i],3)}")
            print(f"-----------------------------")

        return expectation_dict


if __name__ == '__main__':
    hamiltonian_interaction = np.array([[0,-1,-1,-1],
                                        [0,0,-1,-1],
                                        [0,0,0,-1],
                                        [0,0,0,0]])
    qaoa_obj = QAOA(num_elems=4,
                    hamiltonian_type='isling',
                    hamiltonian_interactions=hamiltonian_interaction)
    expectation_dict = qaoa_obj.main()
    
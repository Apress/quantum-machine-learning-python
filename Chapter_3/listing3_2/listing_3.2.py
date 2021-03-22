import cirq
print('circ version',cirq.__version__)
import matplotlib.pyplot as plt
import matplotlib
print('matplotlib version', matplotlib.__version__)


def hadamard_state_measurement(copies):
    # Define a Qubit
    qubit = cirq.GridQubit(0, 0)
    # Create a Circuit in cirq 
    circuit = cirq.Circuit([cirq.H(qubit), cirq.measure(qubit, key='m')])
    print("Circuit Follows")
    print(circuit)
    sim = cirq.Simulator()
    output = sim.run(circuit, repetitions=copies)
    res = output.histogram(key='m')
    prob_0 = dict(res)[0] / copies
    print(prob_0)
    return prob_0


def main(copies_low=100, copies_high=1000):
    print("Coming")
    probability_for_zero_state_trace = []
    copies_trace = []
    for n in range(copies_low, copies_high):
        copies_trace.append(n)
        prob_0 = hadamard_state_measurement(n)
        probability_for_zero_state_trace.append(prob_0)
    plt.plot(copies_trace, probability_for_zero_state_trace)
    plt.xlabel('No of Measurements')
    plt.ylabel("Probability of the State 0")
    plt.title("Convergence Sequence of Probability for State 0")
    plt.show()

if __name__ == '__main__':
    main()

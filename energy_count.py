from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import numpy as np

from qaoa.qaoa import *


def save_qasm(filename, qasm_code):
    with open(filename, "w") as file:
        file.write(qasm_code)

def bitstring_to_array(bitstring):
    bitstring = bitstring.replace(" ", "")
    bitstring = bitstring[::-1]
    return np.array([int(b) for b in bitstring])

def qubo_energy(bitstring, Q):
    x = bitstring_to_array(bitstring)
    return float(x.T @ Q @ x)

def qaoa_energy(Q, beta, gamma, p=3, shots=1000):
    
    ''' возвращается энергия для битстринга с наибольшим количеством coutns  '''

    n = Q.shape[0]

    qasm_code = qaoa_qasm(n=n, p=p, beta=beta, gamma=gamma, qubo=Q)

    save_qasm("qaoa_circuit.qasm", qasm_code)

    qc = QuantumCircuit.from_qasm_file("qaoa_circuit.qasm")

    simulator = AerSimulator()
    compiled_qc = transpile(qc, simulator)

    job = simulator.run(compiled_qc, shots=shots)
    result = job.result()

    counts = result.get_counts()

    best_bitstring = max(counts, key=counts.get) 

    energy = qubo_energy(best_bitstring, Q)

    return energy

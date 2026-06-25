import torch
import pennylane as qml


def mix_layer(n: int, beta):
    for i in range(n):
        arg = torch.remainder(2 * beta, torch.pi)
        qml.RX(arg, wires=i)


def problem_layer(n: int, gamma, qubo):
    for j in range(n):
        for k in range(n):
            if qubo[j][k] != 0:
                arg = torch.remainder(gamma / 2 * qubo[j][k], 2 * torch.pi)

                if j == k:
                    qml.RZ(torch.remainder(-2 * arg, 2 * torch.pi), wires=j)

                else:
                    qml.RZ(-arg, wires=j)
                    qml.RZ(-arg, wires=k)
                    qml.CNOT(wires=[j, k])
                    qml.RZ(arg, wires=k)
                    qml.CNOT(wires=[j, k])


def create_qaoa_circuit(n: int, p: int):
    dev = qml.device("default.qubit", wires=n)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def qaoa_circuit(beta, gamma, qubo):
        # Hadamard gates
        for i in range(n):
            qml.Hadamard(wires=i)

        # QAOA iterations
        for i in range(p):
            mix_layer(n, beta[i])
            problem_layer(n, gamma[i], qubo)

        # Return probabilities of all bitstrings
        return qml.probs(wires=range(n)) # return 2**n array

    return qaoa_circuit


def bitstrings_tensor(n, dtype=torch.float32):
    
    bitstrings = [
        [int(bit) for bit in f"{i:0{n}b}"] for i in range(2 ** n)
    ]

    bits = torch.tensor(bitstrings, dtype=dtype)
    return bits


def qubo_energies_tensor(qubo):
    """
    Calculate xQx
    Output shape: [2**n]
    """
    n = qubo.shape[0]
    bitstrings = bitstrings_tensor(n, dtype=qubo.dtype)

    energies = []

    for x in bitstrings:
        energy = x @ qubo @ x
        energies.append(energy)

    energies = torch.stack(energies)

    return energies


def qaoa_expected_energy(qubo, beta, gamma, qaoa_circuit):

    probs = qaoa_circuit(beta, gamma, qubo)

    energies = qubo_energies_tensor(qubo)
    energies = energies.to(probs.dtype)

    expected_energy = torch.sum(probs * energies)

    return expected_energy


def qaoa_expected_energy_top(qubo, beta, gamma, qaoa_circuit, n=10):

    probs = qaoa_circuit(beta, gamma, qubo)

    energies = qubo_energies_tensor(qubo)
    energies = energies.to(probs.dtype)

    top_indices = torch.topk(probs, n).indices
    top_energies = energies[top_indices]

    return top_energies.mean()
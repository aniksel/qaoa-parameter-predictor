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








def test():
    n = 5
    p = 3

    # qaoa_circuit = create_qaoa_counts_circuit(n, p)
    qaoa_circuit = create_qaoa_circuit(n, p)

    beta = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)
    gamma = torch.tensor([0.5, 0.6, 0.7], dtype=torch.float32)

    qubo = torch.tensor([
        [-2,  1,  0,  0,  1],
        [ 1, -3,  1,  0,  1],
        [ 0,  1, -2,  1,  0],
        [ 0,  0,  1, -2,  1],
        [ 1,  1,  0,  1, -3]
    ], dtype=torch.float32)

    probs = qaoa_circuit(beta, gamma, qubo)
    return probs


# print(test())

# def create_qaoa_counts_circuit(n: int, p: int):
#     dev = qml.device("default.qubit", wires=n)

#     @qml.set_shots(1000)
#     @qml.qnode(dev)
#     def qaoa_circuit(beta, gamma, qubo):
#         for i in range(n):
#             qml.Hadamard(wires=i)

#         for i in range(p):
#             mix_layer(n, beta[i])
#             problem_layer(n, gamma[i], qubo)

#         return qml.counts(wires=range(n))

#     return qaoa_circuit
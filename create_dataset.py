import itertools
import random
import numpy as np
import os
import json


def has_no_isolated_vertices(n, edges):

    degree = [0] * n

    for i, j in edges:
        degree[i] += 1
        degree[j] += 1

    return all(d > 0 for d in degree)


def generate_graph(n=5, m=6):

    all_possible_edges = list(itertools.combinations(range(n), 2)) # create all possible edges

    while True:
        edges = random.sample(all_possible_edges, m) # m - edges

        if has_no_isolated_vertices(n, edges):
            return edges


def graph_to_qubo(n, edges):

    Q = np.zeros((n, n), dtype=int)

    for i, j in edges:
        Q[i][i] -= 1
        Q[j][j] -= 1

        Q[i][j] += 1
        Q[j][i] += 1

    return Q


def create_dataset(num_graphs=20, n=5, m=6, seed=43):
    random.seed(seed)

    dataset = []
    used_graphs = set()

    while len(dataset) < num_graphs:
        edges = generate_graph(n, m)
        graph_curr = tuple(sorted(edges))

        if graph_curr in used_graphs:
            continue

        used_graphs.add(graph_curr)

        dataset.append({
            "matrix": graph_to_qubo(n, edges).tolist(),
            "edges": [list(edge) for edge in edges],
            "n": n
        })

    return dataset


dataset = create_dataset(num_graphs=205, n=5, m=6)

output_dir = "maxcut_qubo_dataset"
os.makedirs(output_dir, exist_ok=True)


def save_qubo_json(item, filepath):
    matrix = item["matrix"]
    edges = item["edges"]
    n = item["n"]

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("{\n")

        f.write('  "matrix": [\n')
        for row_idx, row in enumerate(matrix):
            comma = "," if row_idx < len(matrix) - 1 else ""
            f.write(f"    {json.dumps(row)}{comma}\n")
        f.write("  ],\n")

        f.write(f'  "edges": {json.dumps(edges)},\n')
        f.write(f'  "n": {n}\n')

        f.write("}\n")


for i, item in enumerate(dataset, start=1):
    filepath = os.path.join(output_dir, f"matrix-{i:02d}.json")
    save_qubo_json(item, filepath)

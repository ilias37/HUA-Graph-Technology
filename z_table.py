import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def h(Zi, Zj):
    return np.dot(Zi, Zj)

def custom_algorithm(G, r, epsilon=1e-5, lambda_val=0.1, max_iter=1000):
    n = len(G.nodes())
    Z = np.random.rand(n, r)
    t = 1
    convergence = []

    while True:
        Z0 = Z.copy()

        for edge in G.edges():
            i, j = edge
            eta = np.sqrt(1 / t)
            t += 1

            Z[i] += eta * (1 - h(Z[i], Z[j]) * Z[j] + lambda_val * Z[i])

        convergence.append(np.linalg.norm(Z - Z0, 'fro'))

        if convergence[-1] <= epsilon or t > max_iter:
            break

    return Z, convergence

with open("./p2p-Gnutella08.txt", "r") as file:
    # Skip the first 4 lines
    for _ in range(4):
        file.readline()

    lines = file.readlines()

G = nx.Graph()

# add nodes and edges
for line in lines:
    # Split the line using tabs
    parts = line.strip().split("\t")
    source, target = int(parts[0]), int(parts[1])
    G.add_edge(source, target)

# Graph Visualization of the first
pos_start = nx.spring_layout(G)
nx.draw(G, pos_start, with_labels=True)
plt.title('Starting Graph')
plt.show()

result, convergence = custom_algorithm(G, r=3)

print(result)


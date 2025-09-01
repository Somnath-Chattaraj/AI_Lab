# Unweighted Graph

# Creation
# import csv
# import random

# def generate_binary_adjacency_matrix(n_nodes=12, edge_prob=0.3):
#     matrix = [[0] * n_nodes for _ in range(n_nodes)]
#     for i in range(n_nodes):
#         for j in range(i + 1, n_nodes):
#             if random.random() < edge_prob:
#                 matrix[i][j] = 1
#                 matrix[j][i] = 1  # Ensure symmetry (undirected)
#     return matrix

# def save_matrix_to_csv(matrix, filename='complex_adjacency_matrix.csv'):
#     with open(filename, 'w', newline='') as f:
#         writer = csv.writer(f)
#         writer.writerows(matrix)
#     print(f"Saved to {filename}")

# # Generate and save
# adj_matrix = generate_binary_adjacency_matrix(n_nodes=12, edge_prob=0.35)
# save_matrix_to_csv(adj_matrix)


import csv
import networkx as nx
import matplotlib.pyplot as plt

def load_adjacency_matrix(file_path):
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        matrix = [list(map(int, row)) for row in reader]
    return matrix

def create_graph_from_matrix(matrix, directed=True):
    G = nx.Graph() if directed else nx.Graph()

    n = len(matrix)
    for i in range(n):
        for j in range(len(matrix[i])):
            if matrix[i][j] == 1:
                G.add_edge(i, j)
    return G

def draw_graph(G):
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=1500, font_size=12)
    plt.title("Graph from Adjacency Matrix")
    plt.show()

csv_file = '/content/complex_adjacency_matrix.csv'
adj_matrix = load_adjacency_matrix(csv_file)
graph = create_graph_from_matrix(adj_matrix, directed=True)
draw_graph(graph)



# Weighted Graph

# creation

# import csv
# import random

# def generate_weighted_adjacency_matrix(n_nodes=10, edge_prob=0.4, weight_range=(1, 20)):
#     matrix = [[0] * n_nodes for _ in range(n_nodes)]

#     for i in range(n_nodes):
#         for j in range(i + 1, n_nodes):
#             if random.random() < edge_prob:
#                 weight = random.randint(*weight_range)
#                 matrix[i][j] = weight
#                 matrix[j][i] = weight  # Symmetric for undirected graph
#     return matrix

# def save_weighted_matrix_to_csv(matrix, filename='weighted_adjacency_matrix.csv'):
#     with open(filename, 'w', newline='') as f:
#         writer = csv.writer(f)
#         writer.writerows(matrix)
#     print(f"Weighted adjacency matrix saved as '{filename}'.")

# # Example: Generate a 10-node weighted undirected graph
# matrix = generate_weighted_adjacency_matrix(n_nodes=10, edge_prob=0.4)
# save_weighted_matrix_to_csv(matrix)

import csv
import networkx as nx
import matplotlib.pyplot as plt

def load_weighted_adjacency_matrix(filename):
    """Load numerical weighted adjacency matrix from CSV file."""
    matrix = []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            # Convert each value to float (or zero if empty)
            matrix.append([float(val) if val else 0.0 for val in row])
    return matrix

def create_weighted_graph(matrix):
    """Create NetworkX weighted graph from adjacency matrix."""
    G = nx.Graph()
    n = len(matrix)
    for i in range(n):
        G.add_node(i)
    for i in range(n):
        for j in range(i+1, n):
            weight = matrix[i][j]
            if weight != 0:
                G.add_edge(i, j, weight=weight)
    return G

def draw_weighted_graph(G):
    """Draw weighted graph with edge labels."""
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=700, font_size=12)

    # Get edge weights to label them
    edge_labels = nx.get_edge_attributes(G, 'weight')
    # Format weights to 2 decimals
    edge_labels = {edge: f"{w:.2f}" for edge, w in edge_labels.items()}

    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')

    plt.title("Weighted Graph from Numerical Adjacency Matrix")
    plt.show()

# --- Main ---

filename = '/content/weighted_adjacency_matrix.csv'  # your CSV file here

matrix = load_weighted_adjacency_matrix(filename)
G = create_weighted_graph(matrix)
draw_weighted_graph(G)

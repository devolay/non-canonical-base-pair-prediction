import numpy as np
import networkx as nx
from sklearn.preprocessing import OneHotEncoder


def one_hot_encode_sequence(seq: str) -> np.ndarray:
    """
    One-hot encodes a sequence of amino acids or nucleotides.
    """
    encoder = OneHotEncoder(categories=[["A", "C", "G", "U"]])
    return encoder.fit_transform(np.array(list(seq)).reshape(-1, 1)).toarray()


def get_phosphodiester_bonds_matrix(seq: str) -> np.ndarray:
    """
    Creates a matrix of phosphodiester bonds between nucleotides in a sequence.
    """
    bonds = np.zeros((len(seq), len(seq)), dtype=int)
    for i in range(len(seq) - 1):
        bonds[i, i + 1] = 1
        bonds[i + 1, i] = 1
    return bonds


def one_hot_edges(edges_matrix: np.ndarray, num_classes: int = 15) -> np.ndarray:
    """
    One-hot encodes the edges between nodes in a graph.
    """
    one_hot_matrix = np.zeros((edges_matrix.shape[0], edges_matrix.shape[1], num_classes))
    for edge_class in range(num_classes):
        one_hot_matrix[edges_matrix == edge_class, edge_class] = 1
    return one_hot_matrix


def create_rna_graph(seq: str, pairings_matrix: np.ndarray) -> nx.Graph:
    """
    Creates a NetworkX graph fromsa sequence and an edge matrix.

    The sequence is used to label each node in the graph.
    The edge matrix is used to determine the edges between nodes.
    """
    G = nx.Graph()
    nodes_features = one_hot_encode_sequence(seq)
    for i, node in enumerate(seq):
        G.add_node(i, features=nodes_features[i])

    bond_matrix = get_phosphodiester_bonds_matrix(seq)
    pairings_matrix[pairings_matrix > 0] += 1
    edges_matrix = bond_matrix + pairings_matrix
    edges_features = one_hot_edges(edges_matrix)
    for i in range(len(seq)):
        for j in range(i + 1, len(seq)):
            if edges_matrix[i, j] > 0:
                G.add_edge(i, j, features=edges_features[i, j])

    return G

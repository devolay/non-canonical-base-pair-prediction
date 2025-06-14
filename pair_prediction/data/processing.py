import numpy as np
import networkx as nx
from sklearn.preprocessing import OneHotEncoder


def one_hot_encode_sequence(seq: str) -> np.ndarray:
    """
    One-hot encodes a sequence of amino acids or nucleotides.
    """
    encoder = OneHotEncoder(categories=[["A", "C", "G", "U", "I", "D"]])
    return encoder.fit_transform(np.array(list(seq)).reshape(-1, 1)).toarray().astype(np.float32)


def get_phosphodiester_bonds_matrix(seq: str) -> np.ndarray:
    """
    Creates a matrix of phosphodiester bonds between nucleotides in a sequence.
    """
    bonds = np.zeros((len(seq), len(seq)), dtype=np.float32)
    for i in range(len(seq) - 1):
        bonds[i, i + 1] = 1
        bonds[i + 1, i] = 1
    return bonds


def one_hot_edges(edges_matrix: np.ndarray, num_classes: int = 15) -> np.ndarray:
    """
    One-hot encodes the edges between nodes in a graph.
    """
    one_hot_matrix = np.zeros(
        (edges_matrix.shape[0], edges_matrix.shape[1], num_classes), dtype=np.float32
    )
    for edge_class in range(num_classes):
        one_hot_matrix[edges_matrix == edge_class, edge_class] = 1
    return one_hot_matrix


def create_rna_graph(seq: str, pairings_matrix: np.ndarray, simple: bool = True) -> nx.Graph:
    """
    Creates a NetworkX graph fromsa sequence and an edge matrix.

    The sequence is used to label each node in the graph.
    The edge matrix is used to determine the edges between nodes.

    Args:
        seq (str): RNA sequence
        pairings_matrix (np.ndarray): Matrix containing pair information where:
            - 0: no pair
            - 1: canonical pair
            - >1: non-canonical pair (number indicates the specific type)
        simple (bool): If True, simplifies edge types to just "canonical" and "non-canonical"
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
            edge_type = edges_matrix[i, j]
            features = edges_features[i, j]
            match edge_type:
                case 1:
                    G.add_edge(i, j, features=features, edge_type="phosphodiester", pair_type=0)
                case 2:
                    G.add_edge(i, j, features=features, edge_type="canonical", pair_type=1)
                case edge_type if edge_type > 2:
                    G.add_edge(i, j, features=features, edge_type="non-canonical", pair_type=edge_type-1)
    return G

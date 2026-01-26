import numpy as np
import networkx as nx
import re
from sklearn.preprocessing import OneHotEncoder


def one_hot_encode_sequence(seq: str) -> np.ndarray:
    """
    One-hot encodes a sequence of amino acids or nucleotides.
    """
    encoder = OneHotEncoder(categories=[["A", "C", "G", "U", "D"]])
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


def create_rna_graph(details: list[dict], pairings_matrix: np.ndarray, simple: bool = True) -> nx.Graph:
    """
    Creates a NetworkX graph fromsa sequence and an edge matrix.

    The sequence is used to label each node in the graph.
    The edge matrix is used to determine the edges between nodes.

    Args:
        details (list of dict): List containing residue details with keys:
            - res_number (int): The numeric residue index.
            - chain_id (str): The chain identifier.
            - res_type (str): The single-letter residue code.
            - res_id (str): The numeric part of the residue identifier.
        pairings_matrix (np.ndarray): Matrix containing pair information where:
            - 0: no pair
            - 1: canonical pair
            - >1: non-canonical pair (number indicates the specific type)
        simple (bool): If True, simplifies edge types to just "canonical" and "non-canonical"
    """
    G = nx.Graph()
    seq = "".join([res['res_type'] for res in details])

    nodes_features = one_hot_encode_sequence(seq)
    for i, node in enumerate(details):
        G.add_node(i, features=nodes_features[i])

    bond_matrix = get_phosphodiester_bonds_matrix(seq)
    pairings_matrix[pairings_matrix > 0] += 1

    edges_matrix = bond_matrix + pairings_matrix
    edges_features = one_hot_edges(edges_matrix)
    for i in range(len(seq)):
        res_id_i = int(re.sub("[^0-9]", "", details[i]['res_id']))
        chain_id_i = details[i]['chain_id']
        for j in range(i + 1, len(seq)):
            res_id_j = int(re.sub("[^0-9]", "", details[j]['res_id']))
            chain_id_j = details[j]['chain_id']
            edge_type = edges_matrix[i, j]
            features = edges_features[i, j]
            match edge_type:
                case 1:
                    if chain_id_i != chain_id_j or abs(res_id_i - res_id_j) != 1:
                        continue
                    G.add_edge(i, j, features=features, edge_type="phosphodiester", pair_type=0)
                case 2:
                    G.add_edge(i, j, features=features, edge_type="canonical", pair_type=1)
                case edge_type if edge_type > 2:
                    G.add_edge(i, j, features=features, edge_type="non-canonical", pair_type=edge_type-1)
    return G

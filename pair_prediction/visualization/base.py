import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

def draw_rna_structure(seq: str, amt_matrix: np.ndarray):
    n = len(seq)
    G = nx.Graph()
    for i in range(n):
        G.add_node(i, res_type=seq[i])

    for i in range(n):
        for j in range(i+1, n):
            val = amt_matrix[i][j]
            if val != 0:
                G.add_edge(i, j, weight=val)

    sequence_edges = [(i, i+1) for i in range(n-1)]

    color_map = {
        'A': 'red',
        'C': 'blue',
        'G': 'green',
        'U': 'yellow'
    }
    node_colors = [color_map[G.nodes[i]['res_type']] for i in G.nodes()]

    edge_colors = []
    for (u, v, d) in G.edges(data=True):
        val = d['weight']
        if val == 1:
            edge_colors.append('black')  # canonical
        else:
            edge_colors.append('orange') # non-canonical

    pos = nx.circular_layout(G)

    plt.figure(figsize=(20,20))
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=2000)
    nx.draw_networkx_labels(G, pos, {i: seq[i] for i in G.nodes()}, font_color='black')
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, edgelist=G.edges())
    nx.draw_networkx_edges(G, pos, edgelist=sequence_edges, edge_color='gray', style='dashed')

    plt.title("RNA simplified graph")
    plt.axis('off')
    plt.show()
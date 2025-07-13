import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import to_dense_batch, to_dense_adj

from rinalmo.model.model import RiNALMo
from rinalmo.config import model_config
from rinalmo.data.alphabet import Alphabet
from rinalmo.data.constants import RNA_TOKENS
from rinalmo.model.downstream import SecStructPredictionHead
from pair_prediction.model.residual_block import ResidualBlock1d


class RiNAlmoLinkPredictionModel(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        gnn_channels: list = [64, 64],
        gnn_attention_heads: int = 4,
        cnn_head_embed_dim: int = 64,
        cnn_head_num_blocks: int = 2,
        out_channels: int = 64,
        dropout: float = 0.0,
    ):
        """
        Args:
            in_channels (int): Number of input node features.
            gnn_channels (list of int): Hidden channel sizes for the GNN encoder layers.
                For example, [64, 64] creates a GNN with two layers:
                - First layer: input in_channels, output 64.
                - Second layer: input 64, output 64.
            cnn_channels (list of int): Hidden channel sizes for the CNN encoder layers.
                For example, [64, 64] creates a CNN with two residual blocks:
                - First block: input in_channels, output 64.
                - Second block: input 64, output 64.
            dropout (float): Dropout probability.
        """
        super().__init__()
        self.dropout = dropout
        self.tokenizer = Alphabet()
        self.rinalmo = RiNALMo(model_config("giga"))
        self.pad_idx = self.rinalmo.config['model']['embedding'].padding_idx
        self.rna_indices = torch.tensor([self.tokenizer.get_idx(token) for token in RNA_TOKENS])

        self.gnn_convs = nn.ModuleList()
        self.gnn_convs.append(GATv2Conv(in_channels, int(gnn_channels[0] / gnn_attention_heads), residual=True, heads=gnn_attention_heads))
        for i in range(1, len(gnn_channels)):
            self.gnn_convs.append(GATv2Conv(gnn_channels[i-1], int(gnn_channels[i] / gnn_attention_heads), residual=True, heads=gnn_attention_heads))
        
        self.prediction_head = SecStructPredictionHead(
            embed_dim=gnn_channels[-1],
            num_blocks=cnn_head_num_blocks,
            conv_dim=cnn_head_embed_dim,
            kernel_size=3
        )


    def _load_pretrained_lm_weights(self, pretrained_weights_path: str, freeze_lm: bool = True):
        self.rinalmo.load_state_dict(torch.load(pretrained_weights_path))
        if freeze_lm:
            for param in self.rinalmo.parameters():
                param.requires_grad = False
        
    def forward(self, x: torch.Tensor, tokens: torch.Tensor, edge_index: torch.Tensor):
        """
        Forward pass:
          - Computes node embeddings with the GNN encoder.
          - Computes a global graph representation with the CNN encoder.
          
        Args:
            x (torch.Tensor): Node feature matrix of shape [total_nodes, in_channels].
            edge_index (torch.Tensor): Graph connectivity information.
            batch (torch.Tensor): Batch vector that assigns each node to a graph.
            
        Returns:
            node_embeddings (torch.Tensor): Output from GNN encoder.
            global_reps (torch.Tensor): Global representations from CNN encoder.
        """
        node_embeddings = self.rinalmo(tokens)["representation"]
        nucleotide_mask = torch.isin(tokens, self.rna_indices.to(tokens.device))
        node_embeddings = node_embeddings[nucleotide_mask]

        for conv in self.gnn_convs:
            node_embeddings = conv(node_embeddings, edge_index)
            node_embeddings = F.dropout(node_embeddings, p=self.dropout, training=self.training)
            node_embeddings = F.elu(node_embeddings)
        
        return node_embeddings

        
    def compute_edge_logits(
        self,
        node_embeddings: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute logits for edges by concatenating local node embeddings (src & dst)
        with the corresponding global representation.

        Args:
            node_embeddings (torch.Tensor): Node embeddings [total_nodes, hidden_channels].
            edge_index (torch.Tensor): Tensor of edge indices.
            global_reps (torch.Tensor): Global representations [batch_size, hidden_channels].
            batch (torch.Tensor): Batch vector assigning each node to a graph.

        Returns:
            logits (torch.Tensor): Logits for each edge.
        """
        x_dense, _ = to_dense_batch(node_embeddings, batch)
        logits = self.prediction_head(x_dense)
        edges_dense = to_dense_adj(edge_index, batch)
        return logits[edges_dense.bool()]

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def normalized_laplacian(adjacency):
    adjacency = torch.as_tensor(adjacency, dtype=torch.float32)
    degree = adjacency.sum(dim=1)
    inv_sqrt = torch.zeros_like(degree)
    nonzero = degree > 0
    inv_sqrt[nonzero] = degree[nonzero].pow(-0.5)
    return torch.eye(adjacency.size(0), dtype=adjacency.dtype) - inv_sqrt[:, None] * adjacency * inv_sqrt[None, :]


def wavelet_bases(adjacency, scales, directed=False):
    adjacency = torch.as_tensor(adjacency, dtype=torch.float32)
    graphs = [adjacency, adjacency.t()] if directed else [adjacency]
    bases = []
    inverses = []
    for graph in graphs:
        laplacian = normalized_laplacian(graph)
        bases.append(torch.stack([torch.linalg.matrix_exp(float(scale) * laplacian) for scale in scales]))
        inverses.append(torch.stack([torch.linalg.matrix_exp(-float(scale) * laplacian) for scale in scales]))
    return torch.stack(bases), torch.stack(inverses)


class AdaptiveWaveletBlock(nn.Module):
    def __init__(self, adjacency, in_channels, out_channels, scales, directed=False, attention_dim=None):
        super().__init__()
        self.num_nodes = int(np.asarray(adjacency).shape[0])
        self.num_scales = len(scales)
        self.num_directions = 2 if directed else 1
        attention_dim = attention_dim or out_channels
        bases, inverses = wavelet_bases(adjacency, scales, directed)
        self.register_buffer("wavelet_basis", bases)
        self.register_buffer("wavelet_inverse", inverses)
        self.filters = nn.Parameter(torch.ones(self.num_directions, self.num_scales, self.num_nodes))
        self.projections = nn.ModuleList([nn.Linear(in_channels, out_channels, bias=False) for _ in scales])
        self.direction_projection = nn.Linear(self.num_directions * out_channels, out_channels, bias=False) if directed else None
        self.query_projection = nn.Linear(in_channels, attention_dim, bias=False)
        self.value_projection = nn.Linear(out_channels, attention_dim, bias=False)

    def branch_operators(self, correction, shift_weight):
        diagonal = torch.diag_embed(self.filters)
        operators = self.wavelet_basis @ diagonal @ self.wavelet_inverse
        return F.relu(operators + shift_weight * correction[None, None, :, :])

    def forward(self, inputs, correction, shift_weight, return_attention=False):
        operators = self.branch_operators(correction, shift_weight)
        branches = []
        for scale_index, projection in enumerate(self.projections):
            directional = []
            for direction_index in range(self.num_directions):
                propagated = torch.einsum("nm,btmc->btnc", operators[direction_index, scale_index], inputs)
                directional.append(projection(propagated))
            branch = directional[0] if self.num_directions == 1 else self.direction_projection(torch.cat(directional, dim=-1))
            branches.append(F.relu(branch))
        branches = torch.stack(branches, dim=3)
        queries = self.query_projection(inputs)
        values = self.value_projection(branches)
        scores = F.cosine_similarity(queries.unsqueeze(3), values, dim=-1, eps=1e-8)
        attention = torch.softmax(scores, dim=3)
        output = torch.sum(attention.unsqueeze(-1) * branches, dim=3)
        if return_attention:
            return output, attention
        return output


class AWN(nn.Module):
    def __init__(
        self,
        adjacency,
        input_channels=1,
        hidden_channels=32,
        output_channels=1,
        history_length=12,
        prediction_length=12,
        scales=None,
        rank=30,
        shift_weight=0.01,
        graph_layers=3,
        recurrent_layers=2,
        directed=False,
    ):
        super().__init__()
        if scales is None:
            scales = [0.1 + 0.2 * index for index in range(20)]
        self.num_nodes = int(np.asarray(adjacency).shape[0])
        self.history_length = history_length
        self.prediction_length = prediction_length
        self.shift_weight = shift_weight
        self.blocks = nn.ModuleList()
        self.residual_projections = nn.ModuleList()
        for layer_index in range(graph_layers):
            in_channels = input_channels if layer_index == 0 else hidden_channels
            self.blocks.append(AdaptiveWaveletBlock(adjacency, in_channels, hidden_channels, scales, directed))
            self.residual_projections.append(nn.Linear(input_channels, hidden_channels))
        self.low_rank_left = nn.Parameter(torch.empty(self.num_nodes, rank))
        self.low_rank_right = nn.Parameter(torch.empty(rank, self.num_nodes))
        self.recurrent = nn.GRU(hidden_channels, hidden_channels, recurrent_layers, batch_first=True)
        self.readout = nn.Linear(hidden_channels, output_channels)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.low_rank_left)
        nn.init.xavier_uniform_(self.low_rank_right)

    def encode_context(self, inputs, return_attention=False):
        if inputs.ndim != 4:
            raise ValueError("inputs must have shape (batch, history, nodes, features)")
        if inputs.size(1) != self.history_length or inputs.size(2) != self.num_nodes:
            raise ValueError("input history length or node count does not match the model")
        correction = self.low_rank_left @ self.low_rank_right
        original = inputs
        hidden = inputs
        attentions = []
        for block, residual_projection in zip(self.blocks, self.residual_projections):
            if return_attention:
                hidden, attention = block(hidden, correction, self.shift_weight, True)
                attentions.append(attention)
            else:
                hidden = block(hidden, correction, self.shift_weight)
            hidden = hidden + residual_projection(original)
        batch_size = hidden.size(0)
        sequence = hidden.permute(0, 2, 1, 3).reshape(batch_size * self.num_nodes, self.history_length, -1)
        outputs, state = self.recurrent(sequence)
        decoder_input = outputs[:, -1:, :]
        predictions = []
        for _ in range(self.prediction_length):
            decoder_output, state = self.recurrent(decoder_input, state)
            predictions.append(self.readout(decoder_output[:, 0, :]))
        prediction = torch.stack(predictions, dim=1)
        prediction = prediction.reshape(batch_size, self.num_nodes, self.prediction_length, -1).permute(0, 2, 1, 3)
        if return_attention:
            return prediction, torch.stack(attentions, dim=1)
        return prediction

    def forward(self, inputs, return_attention=False):
        return self.encode_context(inputs, return_attention)

from itertools import combinations
import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomLoss(nn.Module):
    def __init__(self, sim_matrix, device):
        super(CustomLoss, self).__init__()
        self.sim_matrix = torch.from_numpy(sim_matrix).to(device)

    def forward(self, inputs, indices):
        inputs_combn = combinations(inputs, 2)
        indices_combn = combinations(range(len(inputs)), 2)
        loss = 0.0
        for inp, idx in zip(inputs_combn, indices_combn):
            distance = F.pairwise_distance(inp[0], inp[1], 2)
            similarity = (self.sim_matrix[indices[:, idx[0]], indices[:, idx[1]]])
            loss += (distance - similarity) ** 2
        return loss.mean()

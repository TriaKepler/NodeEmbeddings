import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, emb_size, classes_num):
        super().__init__()
        self.embeddings = nn.Parameter(torch.randn(*emb_size))
        self.fc = nn.Linear(emb_size[1], classes_num)

    def forward(self, inputs):
        emb_list = [(torch.index_select(self.embeddings, 0, inp)) for inp in inputs]
        emb_tensor = torch.stack(emb_list, dim=1)
        return emb_list, emb_tensor

    def feed_fc(self, inputs):
        out = self.fc(inputs)
        out = F.log_softmax(out, dim=1)
        return out

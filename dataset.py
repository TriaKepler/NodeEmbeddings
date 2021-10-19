from numpy.random import choice as rnd_choice
import random as rnd
from torch.utils.data import Dataset

class GraphDataset(Dataset):
    def __init__(self, graph, size, labels):
        self.nodes_num = len(graph.nodes)
        self.graph = graph
        self.edges = [[vi, vj] for vi, vj in self.graph.edges]
        self.len_edges = len(self.edges)
        self.size = size
        self.data = np.zeros((len(self), 4)).astype(np.int64)
        self.main_labels = labels
        self.labels = np.zeros((len(self), 4)).astype(np.int64)
        self.prepare()

    def __len__(self):
        return self.size * self.len_edges

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def prepare(self):
        for s in range(self.size):
            rnd.shuffle(self.edges)
            for i, edge in enumerate(tqdm(self.edges)):
                n1, n2 = edge
                n3, n4 = rnd_choice(self.nodes_num, 2, replace=False)
                for idx, node in enumerate([n1, n2, n3, n4]):
                    self.data[i + s * self.len_edges][idx] = node
                    self.labels[i + s * self.len_edges][idx] = self.main_labels[node]

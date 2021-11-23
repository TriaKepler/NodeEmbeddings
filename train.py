import torch
import torch.nn.functional as F
import numpy as np
from scipy.spatial import distance_matrix
from node_emb.dataset import GraphDataset
from node_emb.loss import CustomLoss
from node_emb.model import Model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(graph, feat, labels, sim_matrix, depth=256, classes_num=None, size=5, batch_size=64, num_epochs=8, learning_rate=0.01, verbose=False, misc=False):
    n_nodes = len(graph.nodes)
    train_dataset = GraphDataset(graph=graph, size=size, labels=labels, misc=misc)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    model = Model((n_nodes, depth), classes_num).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = CustomLoss(sim_matrix, device).to(device)

    total_step = len(train_loader)
    for epoch in range(num_epochs):
        if verbose:
            print("~"*50)
            emb_numpy = torch.sigmoid(model.embeddings).data.cpu().detach().numpy()
            euc_dist = distance_matrix(emb_numpy, emb_numpy, p=2)
            print("total error:", np.sum((euc_dist - sim_matrix) ** 2) / n_nodes ** 2)
            # print(F.pairwise_distance(,torch.sigmoid(model.embeddings)))
            # print(torch.sum((F.pairwise_distance(torch.tanh(model.embeddings),torch.tanh(model.embeddings)) - (torch.from_numpy(sim_matrix)).to(device))**2).item())
            print("~"*50)
        for batch_idx, out in enumerate(train_loader):
            quad, label = out[0].to(device), out[1].to(device)
            n1, n2, n3, n4 = quad[:, 0], quad[:, 1], quad[:, 2], quad[:, 3]
            emb_list, emb_tensor = model([n1, n2, n3, n4])
            idx = torch.zeros(1, emb_tensor.size(0)).long()
            # idx = torch.randint(0, 4, (1, emb_tensor.size(0)))
            j = model.feed_fc((emb_tensor[torch.arange(emb_tensor.size(0)), idx].squeeze(0)))
            label_ = label[torch.arange(emb_tensor.size(0)), idx].squeeze(0)
            loss2 = F.nll_loss(j, label_.squeeze(-1))
            loss1 = criterion(emb_list, quad)
            loss = 0.5 * loss1 + 0.5 * loss2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (batch_idx + 1) % 10 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, batch_idx + 1, total_step, loss.item()))

    return model.embeddings.data.cpu().detach().numpy()

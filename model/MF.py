import torch
import torch.nn as nn
import torch.nn.functional as F


class model_MF(nn.Module):
    def __init__(self, num_users, num_items, embedding_size):
        super(model_MF, self).__init__()
        self.user_emb = nn.Embedding(num_users, embedding_size)
        self.item_emb = nn.Embedding(num_items, embedding_size)

        self.user_emb.weight.data = torch.nn.init.xavier_uniform_(self.user_emb.weight.data)
        self.item_emb.weight.data = torch.nn.init.xavier_uniform_(self.item_emb.weight.data)

    def forward(self, u_id, i_id):
        U = self.user_emb(u_id)
        I = self.item_emb(i_id)
        return (U * I).sum(1)


class Controller(nn.Module):
    def __init__(self, dim1, device):
        super(Controller, self).__init__()

        self.linear1 = nn.Linear(dim1, 6, bias=True).to(device)
        self.linear2 = nn.Linear(6, 1, bias=False).to(device)

    def forward(self, x):
        z1 = torch.relu(self.linear1(x))
        res = F.sigmoid(self.linear2(z1))
        # res = F.softplus(self.linear2(z1))

        return res

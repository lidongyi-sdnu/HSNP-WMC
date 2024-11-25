import torch
import torch.nn as nn
import torch.nn.functional as F
from layer import AttentionLayer


class Att_gru(nn.Module):

    def __init__(self, nfeat, nhid, nclass, input_D, output_D, dropout, alpha, nheads):

        super(Att_gru, self).__init__()
        self.dropout = dropout

        self.attentions = [
            AttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True)
            for _ in range(nheads)
        ]  
        
        for i, attention in enumerate(self.attentions):
            self.add_module("attention_{}".format(i), attention)

        self.out_att = AttentionLayer(
            nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False
        )  
        self.weight_Dz = nn.Parameter(torch.zeros(size=(input_D, output_D)))
        nn.init.xavier_uniform_(self.weight_Dz.data)

        self.weight_Uz = nn.Parameter(torch.zeros(size=(input_D, output_D)))
        nn.init.xavier_uniform_(self.weight_Uz.data)

        self.weight_Dr = nn.Parameter(torch.zeros(size=(input_D, output_D)))
        nn.init.xavier_uniform_(self.weight_Dr.data)

        self.weight_Ur = nn.Parameter(torch.zeros(size=(input_D, output_D)))
        nn.init.xavier_uniform_(self.weight_Ur.data)

        self.weight_W = nn.Parameter(torch.zeros(size=(input_D, output_D)))
        nn.init.xavier_uniform_(self.weight_W.data)

        self.weight_U = nn.Parameter(torch.zeros(size=(input_D, output_D)))
        nn.init.xavier_uniform_(self.weight_U.data)

        self.W = nn.Parameter(torch.zeros(size=(nfeat, 8 * nhid)))
        nn.init.xavier_uniform_(self.W.data)

        self.b_1 = nn.Parameter(torch.zeros(size=(input_D, 8 * nhid)))
        nn.init.xavier_uniform_(self.b_1.data)

        self.b_2 = nn.Parameter(torch.zeros(size=(input_D, 8 * nhid)))
        nn.init.xavier_uniform_(self.b_2.data)

    def forward(self, x, adj):
        y1 = F.dropout(x, self.dropout, training=self.training)
        y = torch.cat([att(y1, adj) for att in self.attentions], dim=1)

        y2 = torch.matmul(y1, self.W)
        i_r = torch.matmul(self.weight_Dz, y2)
        h_r = torch.matmul(self.weight_Uz, y)
        i_z = torch.matmul(self.weight_Dr, y2)
        h_z = torch.matmul(self.weight_Ur, y)
        w = torch.matmul(self.weight_W, y2)

        reset_gate = torch.sigmoid(i_r + h_r)
        update_gate = torch.sigmoid(i_z + h_z)
        new_state = torch.matmul(self.weight_U, (torch.mul(reset_gate, y)))
        new_state = torch.tanh(w + new_state)

        # att_score = att_score.view(-1, 1)
        # update_gate = att_score * update_gate
        y = torch.mul((1. - update_gate), y) + torch.mul(update_gate, new_state)
        # y = torch.add(y,x)
        y = F.dropout(y, self.dropout, training=self.training)
        y = F.elu(self.out_att(y, adj))
        return F.log_softmax(y, dim=1)








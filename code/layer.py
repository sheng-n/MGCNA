import torch.nn as nn
from torch_geometric.nn import GCNConv
import torch
import torch.nn.functional as F
from parms_setting import settings
args = settings()


class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(Attention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        # print("w: ", w.shape)
        beta = torch.softmax(w, dim=1)
        # print(" beta: ",beta.shape)
        return (beta * z).sum(1), beta

class MGCNA(nn.Module):
    def __init__(self, feature, hidden1, hidden2, decoder1):
        """
        :param args: Arguments object.
        """
        super(MGCNA, self).__init__()
        self.gcn_x1_s = GCNConv(feature, hidden1)  # miRNA sequence view encoder first layer
        self.gcn_x1_g = GCNConv(feature, hidden1)  # miRNA functional view encoder first layer
        self.gcn_x1_r = GCNConv(feature, hidden1)  # miRNA drug view encoder first layer

        self.gcn_x2_s = GCNConv(hidden1, hidden2)  # miRNA sequence view encoder second layer
        self.gcn_x2_g = GCNConv(hidden1, hidden2)  # miRNA functional view encoder second layer
        self.gcn_x2_r = GCNConv(hidden1, hidden2)  # miRNA drug view encoder second layer


        self.gcn_y1_f = GCNConv(feature, hidden1)  # drug fingerprint view encoder first layer
        self.gcn_y1_g = GCNConv(feature, hidden1) # drug gene view encoder first layer
        self.gcn_y1_m = GCNConv(feature, hidden1)  # drug miRNA view encoder first layer

        self.gcn_y2_f = GCNConv(hidden1, hidden2) # drug fingerprint view encoder second layer
        self.gcn_y2_g = GCNConv(hidden1, hidden2) # drug gene view encoder second layer
        self.gcn_y2_m = GCNConv(hidden1, hidden2) # drug miRNA view encoder second layer

        # attention
        self.attention_m = Attention(hidden2)
        self.attention_d = Attention(hidden2)

        # decoder
        self.decoder1 = nn.Linear(hidden2 * 4, decoder1)
        self.decoder2 = nn.Linear(decoder1, 1)


    def forward(self, data, idx):

        torch.manual_seed(1)
        x_m = torch.randn(args.miRNA_number, args.dimensions)
        x_d = torch.randn(args.drug_number, args.dimensions)

        # miRNA encoder
        x_m_s1 = torch.relu(self.gcn_x1_s(x_m.cuda(), data['mm_s']['edges'].cuda())) # first miRNA sequence encoder
        x_m_s2 = torch.relu(self.gcn_x2_s(x_m_s1, data['mm_s']['edges'].cuda()))     # second miRNA sequence encoder

        # x_m_g1 = torch.relu(self.gcn_x1_g(x_m.cuda(), data['mm_g']['edges'].cuda())) # first miRNA functional encoder
        # x_m_g2 = torch.relu(self.gcn_x2_g(x_m_g1, data['mm_g']['edges'].cuda()))     # second miRNA functional encoder
        # #
        x_m_r1 = torch.relu(self.gcn_x1_r(x_m.cuda(), data['mm_r']['edges'].cuda())) # first miRNA drug functional encoder
        x_m_r2 = torch.relu(self.gcn_x2_r(x_m_r1, data['mm_r']['edges'].cuda()))     # second miRNA drug functional encoder


        # drug encoder
        y_d_f1 = torch.relu(self.gcn_y1_f(x_d.cuda(), data['dd_f']['edges'].cuda())) # first drug fingerprint encoder
        y_d_f2 = torch.relu(self.gcn_y2_f(y_d_f1, data['dd_f']['edges'].cuda()))     # second drug fingerprint encoder

        y_d_g1 = torch.relu(self.gcn_y1_g(x_d.cuda(), data['dd_g']['edges'].cuda())) # first drug gene encoder
        y_d_g2 = torch.relu(self.gcn_y2_g(y_d_g1, data['dd_g']['edges'].cuda()))     # second drug gene encoder
        #
        y_d_m1 = torch.relu(self.gcn_y1_m(x_d.cuda(), data['dd_m']['edges'].cuda())) # first drug gene encoder
        y_d_m2 = torch.relu(self.gcn_y2_m(y_d_m1, data['dd_m']['edges'].cuda()))     # second drug gene encoder

        # attention
        x_m = torch.stack([x_m_s2, x_m_r2], dim=1)
        x_m, att_m = self.attention_m(x_m)

        y_d = torch.stack([y_d_f2, y_d_g2, y_d_m2], dim=1)  #y_d_g2,
        y_d, att_d = self.attention_d(y_d)


        entity1 = x_m[idx[0]]
        entity2 = y_d[idx[1]]


        # multi-relationship modelling decoder
        add = entity1 + entity2
        product = entity1 * entity2
        concatenate = torch.cat((entity1, entity2), dim=1)

        feature = torch.cat((add, product, concatenate), dim=1)

        log1 = F.relu(self.decoder1(feature))
        log = self.decoder2(log1)

        return log













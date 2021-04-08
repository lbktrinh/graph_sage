import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def get_neigh_matrix(samp_neighs, unique_nodes_dict, aggregator_type):
    mask = torch.zeros(len(samp_neighs), len(unique_nodes_dict))
    column_indices = [unique_nodes_dict[n] for samp_neigh in samp_neighs for n in samp_neigh]
    row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
    mask[row_indices, column_indices] = 1
    if aggregator_type == 'mean':
        num_neigh = mask.sum(1, keepdim=True)
        mask = mask.div(num_neigh)
        return mask
    else:
        return mask


class MeanAggregator(nn.Module):
    def __init__(self):
        super(MeanAggregator, self).__init__()

    def forward(self, self_feats, features_neighs, neigh_matrix):

        neigh_feats = neigh_matrix.mm(features_neighs)
        # neigh_feats = torch.matmul(neigh_matrix, features_neighs)
        return neigh_feats


class AttentionAggregator(nn.Module):

    # Aggregates a node's embeddings using mean of neighbors' embeddings
    def __init__(self, feature_dim):
        super(AttentionAggregator, self).__init__()

        self.leak_relu = nn.LeakyReLU(0.2)

        self.a = nn.Parameter(torch.FloatTensor(2 * feature_dim, 1))
        nn.init.xavier_uniform_(self.a)

    def forward(self, self_feats, features_neighs, neigh_matrix):

        soft_max = nn.Softmax(dim=0)
        neigh_feats_attention = torch.zeros_like(self_feats)
        for i in range(neigh_matrix.size()[0]):
            feat_neigh_node = features_neighs[neigh_matrix[i].bool()]
            attention = self.leak_relu((torch.cat([self_feats[i].expand_as(feat_neigh_node), feat_neigh_node], dim=1)).mm(self.a))
            attention = soft_max(attention)
            neigh_feats_attention[i] = torch.sum(torch.mul(feat_neigh_node, attention), dim=0)

        return neigh_feats_attention


class Encoder(nn.Module):

    # Encodes a node's using 'convolutional' GraphSage approach
    def __init__(self, feature_dim, embed_dim):
        super(Encoder, self).__init__()

        self.feature_dim = feature_dim
        self.embed_dim = embed_dim

        self.weight = nn.Parameter(torch.FloatTensor(2 * feature_dim, embed_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, neigh_feats, self_feats):

        combined = torch.cat([self_feats, neigh_feats], dim=1)
        combined = F.relu(combined.mm(self.weight))

        return combined


class SupervisedGraphSage(nn.Module):

    def __init__(self, num_classes, feat_dim=16, hid_dim=(128, 128), k_deep=1, aggregator='mean', nheads=(1, 1), dense_dim=512):
        super(SupervisedGraphSage, self).__init__()

        self.num_classes = num_classes
        self.k_deep = k_deep
        self.aggregator = aggregator
        if k_deep == 1:
            if aggregator == 'mean':
                self.agg1 = [MeanAggregator() for _ in range(nheads[0])]
            if aggregator == 'attention':
                self.agg1 = [AttentionAggregator(feature_dim=feat_dim) for _ in range(nheads[0])]
            for i, agg1 in enumerate(self.agg1):
                self.add_module('agg1_{}'.format(i), agg1)

            self.enc1 = [Encoder(feature_dim=feat_dim, embed_dim=hid_dim[0]) for _ in range(nheads[0])]
            for i, enc1 in enumerate(self.enc1):
                self.add_module('enc1_{}'.format(i), enc1)

            self.weight = nn.Parameter(torch.FloatTensor(nheads[0] * hid_dim[0], num_classes))
        if k_deep == 2:
            if aggregator == 'mean':
                self.agg1 = [MeanAggregator() for _ in range(nheads[0])]
                self.agg2 = [MeanAggregator() for _ in range(nheads[1])]
            if aggregator == 'attention':
                self.agg1 = [AttentionAggregator(feature_dim=feat_dim) for _ in range(nheads[0])]
                self.agg2 = [AttentionAggregator(feature_dim=nheads[0] * hid_dim[0]) for _ in range(nheads[1])]
            for i, agg1 in enumerate(self.agg1):
                self.add_module('agg1_{}'.format(i), agg1)
            for i, agg2 in enumerate(self.agg2):
                self.add_module('agg2_{}'.format(i), agg2)

            self.enc1 = [Encoder(feature_dim=feat_dim, embed_dim=hid_dim[0]) for _ in range(nheads[0])]
            self.enc2 = [Encoder(feature_dim=nheads[0] * hid_dim[0], embed_dim=hid_dim[1]) for _ in range(nheads[1])]
            for i, enc1 in enumerate(self.enc1):
                self.add_module('enc1_{}'.format(i), enc1)
            for i, enc2 in enumerate(self.enc2):
                self.add_module('enc2_{}'.format(i), enc2)

            self.weight = nn.Parameter(torch.FloatTensor(nheads[1] * hid_dim[1], num_classes))

        nn.init.xavier_uniform_(self.weight)

    def forward(self, features, adj_lists, batch_nodes):
        self_feats_of_batch_nodes = torch.FloatTensor(features[np.array(batch_nodes)])

        if self.k_deep == 1:
            set_neigh_of_batch_nodes = [adj_lists[node] for node in batch_nodes]
            unique_neigh_of_batch_nodes_list = list(set.union(*set_neigh_of_batch_nodes))
            unique_neigh_of_batch_nodes_list_dict = {n: i for i, n in enumerate(unique_neigh_of_batch_nodes_list)}
            self_feats_neigh_of_batch_nodes = torch.FloatTensor(features[np.array(unique_neigh_of_batch_nodes_list)])

            neigh_of_batch_nodes_matrix = get_neigh_matrix(set_neigh_of_batch_nodes, unique_neigh_of_batch_nodes_list_dict, self.aggregator)
            output_enc1_batch_nodes_nheads = []

            for agg1, enc1 in zip(self.agg1, self.enc1):

                agg1_batch_nodes = agg1(self_feats_of_batch_nodes, self_feats_neigh_of_batch_nodes, neigh_of_batch_nodes_matrix)  # dim=15
                enc1_batch_nodes = enc1(agg1_batch_nodes, self_feats_of_batch_nodes)  # dim=32
                output_enc1_batch_nodes_nheads.append(enc1_batch_nodes)

            # out
            enc_batch_nodes = torch.cat(output_enc1_batch_nodes_nheads, dim=1)

        if self.k_deep == 2:
            set_neigh_of_batch_nodes = [adj_lists[node] for node in batch_nodes]
            unique_neigh_of_batch_nodes_list = list(set.union(*set_neigh_of_batch_nodes))
            unique_neigh_of_batch_nodes_list_dict = {n: i for i, n in enumerate(unique_neigh_of_batch_nodes_list)}
            self_feats_neigh_of_batch_nodes = torch.FloatTensor(features[np.array(unique_neigh_of_batch_nodes_list)])

            set_neigh_of_neigh_of_batch_nodes = [adj_lists[node] for node in unique_neigh_of_batch_nodes_list]
            unique_neigh_of_neigh_of_batch_nodes_list = list(set.union(*set_neigh_of_neigh_of_batch_nodes))
            unique_neigh_of_neigh_of_batch_nodes_list_dict = {n: i for i, n in enumerate(unique_neigh_of_neigh_of_batch_nodes_list)}
            self_feats_neigh_of_neigh_of_batch_nodes = torch.FloatTensor(features[np.array(unique_neigh_of_neigh_of_batch_nodes_list)])

            neigh_of_batch_nodes_matrix = get_neigh_matrix(set_neigh_of_batch_nodes, unique_neigh_of_batch_nodes_list_dict, self.aggregator)
            neigh_of_neigh_of_batch_nodes_matrix = get_neigh_matrix(set_neigh_of_neigh_of_batch_nodes, unique_neigh_of_neigh_of_batch_nodes_list_dict, self.aggregator)

            # aggregator 1
            output_enc1_batch_nodes_nheads = []
            output_enc1_neigh_of_batch_nodes_nheads = []
            for agg1, enc1 in zip(self.agg1, self.enc1):
                agg1_batch_nodes = agg1(self_feats_of_batch_nodes, self_feats_neigh_of_batch_nodes, neigh_of_batch_nodes_matrix)  # dim=15
                enc1_batch_nodes = enc1(agg1_batch_nodes, self_feats_of_batch_nodes)  # dim=32

                agg1_neigh_of_batch_nodes = agg1(self_feats_neigh_of_batch_nodes, self_feats_neigh_of_neigh_of_batch_nodes, neigh_of_neigh_of_batch_nodes_matrix)  # dim=15
                enc1_neigh_of_batch_nodes = enc1(agg1_neigh_of_batch_nodes, self_feats_neigh_of_batch_nodes)  # dim=32

                output_enc1_batch_nodes_nheads.append(enc1_batch_nodes)
                output_enc1_neigh_of_batch_nodes_nheads.append(enc1_neigh_of_batch_nodes)

            output_enc1_batch_nodes_nheads = torch.cat(output_enc1_batch_nodes_nheads, dim=1)
            output_enc1_neigh_of_batch_nodes_nheads = torch.cat(output_enc1_neigh_of_batch_nodes_nheads, dim=1)

            # aggregator 2
            output_enc2_batch_nodes_nheads = []
            for agg2, enc2 in zip(self.agg2, self.enc2):
                agg2_batch_nodes = agg2(output_enc1_batch_nodes_nheads, output_enc1_neigh_of_batch_nodes_nheads, neigh_of_batch_nodes_matrix)  # dim=32
                enc2_batch_nodes = enc2(agg2_batch_nodes, output_enc1_batch_nodes_nheads)  # dim=64

                output_enc2_batch_nodes_nheads.append(enc2_batch_nodes)
            # out
            enc_batch_nodes = torch.cat(output_enc2_batch_nodes_nheads, dim=1)

        # output class score
        scores = enc_batch_nodes.mm(self.weight)

        return scores

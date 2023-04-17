"""
Modified from https://github.com/JavierAntoran/Bayesian-Neural-Networks
"""
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np


def sample_weights(w_mu, b_mu, w_p, b_p):
    """
    Quick method for sampling weights and exporting weights
    """
    eps_w = w_mu.data.new(w_mu.size()).normal_()
    # sample parameters
    std_w = 1e-6 + F.softplus(w_p, beta=1, threshold=20)
    W = w_mu + 1 * std_w * eps_w

    if b_mu is not None:
        std_b = 1e-6 + F.softplus(b_p, beta=1, threshold=20)
        eps_b = b_mu.new(b_mu.size()).normal_()
        b = b_mu + 1 * std_b * eps_b
    else:
        b = None

    return W, b


def kld_cost(mu_p, sig_p, mu_q, sig_q):
    # https://arxiv.org/abs/1312.6114 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kld = 0.5 * (2 * torch.log(sig_p / sig_q) - 1 +
                 (sig_q / sig_p).pow(2) + ((mu_p - mu_q) / sig_p).pow(2)
                 ).sum()
    return kld


class BayesLinear_local_reparam(nn.Module):
    """
    Linear Layer where activations are sampled from a fully factorised normal which is given by aggregating
        the moments of each weight's normal distribution.
    The KL divergence is obtained in closed form.
    Only works with gaussian priors.
    """

    def __init__(self, n_in, n_out, prior_sig):
        super(BayesLinear_local_reparam, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.prior_sig = prior_sig

        # Learnable parameters
        self.w_mu = nn.Parameter(torch.empty(self.n_in, self.n_out))
        self.w_p = nn.Parameter(torch.empty(self.n_in, self.n_out))

        self.b_mu = nn.Parameter(torch.empty(self.n_out))
        self.b_p = nn.Parameter(torch.empty(self.n_out))

    def initialize(self):
        nn.init.uniform_(self.w_mu, -0.1, 0.1)
        nn.init.uniform_(self.b_mu, -0.1, 0.1)

        nn.init.uniform_(self.w_p, -3, -2)
        nn.init.uniform_(self.b_p, -3, -2)

    def forward(self, x):

        # calculate std
        std_w = 1e-6 + F.softplus(self.w_p, beta=1, threshold=20)
        std_b = 1e-6 + F.softplus(self.b_p, beta=1, threshold=20)

        act_w_mu = torch.mm(x, self.w_mu)  # self.W_mu + std_w * eps_W
        # why not using the function defined in the original paper?
        act_w_std = torch.sqrt(torch.mm(x.pow(2), std_w.pow(2)))

        eps_w = torch.empty_like(act_w_std).normal_(mean=0, std=1)
        eps_b = torch.empty_like(std_b).normal_(mean=0, std=1)

        act_w_out = act_w_mu + act_w_std * eps_w  # (batch_size, n_output)
        act_b_out = self.b_mu + std_b * eps_b

        output = act_w_out + act_b_out.unsqueeze(0).expand(x.shape[0], -1)

        kld = kld_cost(mu_p=0, sig_p=self.prior_sig, mu_q=self.w_mu, sig_q=std_w) + \
            kld_cost(mu_p=0, sig_p=0.1, mu_q=self.b_mu, sig_q=std_b)

        return output, kld, 0


class RGCNPredictor_BBP(nn.Module):
    def __init__(self, node_feature_dim, gcn_hidden_dim=[256, 256, 256, 256], linear_hidden_dim=[256, 256],
                 output_dim=1, drop_ratio=0.5, graph_pooling='mean', prior_sig=0.1, num_relations=4):
        super(RGCNPredictor_BBP, self).__init__()

        self.prior_sig = prior_sig
        if num_relations == 1:
            self.node_embedding = nn.Linear(node_feature_dim, gcn_hidden_dim[0])
        else:
            self.node_embedding = AtomEncoder(gcn_hidden_dim[0])

        self.gcn_layers_dim = [gcn_hidden_dim[0]] + gcn_hidden_dim
        self.gcn_layers = nn.ModuleList()
        for layer_idx, (in_dim, out_dim) in enumerate(zip(self.gcn_layers_dim[:-1], self.gcn_layers_dim[1:])):
            self.gcn_layers.append(RGCNConv(in_dim, out_dim, num_relations))

        if graph_pooling == 'mean':
            self.pooling = global_mean_pool

        self.dropout = nn.Dropout(p=drop_ratio)
        if graph_pooling == 'set2set':
            self.linear_layers_dim = [gcn_hidden_dim[-1] * 2] + linear_hidden_dim + [output_dim]
        else:
            self.linear_layers_dim = [gcn_hidden_dim[-1]] + linear_hidden_dim + [output_dim]

        self.bayes_linear_layers = nn.ModuleList()
        for layer_idx, (in_dim, out_dim) in enumerate(zip(self.linear_layers_dim[:-1], self.linear_layers_dim[1:])):
            if layer_idx == 0 or layer_idx == 1:
                self.bayes_linear_layers.append(nn.Linear(in_dim, out_dim))
            else:
                self.bayes_linear_layers.append(BayesLinear_local_reparam(in_dim, out_dim, self.prior_sig))
        self.output_dim = output_dim

    def graph_repsenetation(self, x, edge_index, edge_type, batch):

        features = self.node_embedding(x)
        for layer_idx, layer in enumerate(self.gcn_layers):
            features = layer(features, edge_index, edge_type)
            if layer_idx == len(self.gcn_layers) - 1:
                pass
            else:
                features = self.dropout(F.relu(features))
        features = self.pooling(features, batch)

        return features

    def forward(self, batched_data, sample=False):
        if batched_data.edge_attr is None:
            edge_type = torch.zeros(batched_data.edge_index.shape[1], dtype=torch.long).to(
                batched_data.edge_index.device)
        else:
            edge_type = batched_data.edge_attr
        if batched_data.x is None:
            x = degree(batched_data.edge_index[0], batched_data.num_nodes).long()
        else:
            x = batched_data.x
        edge_index, batch = batched_data.edge_index, batched_data.batch

        features = self.graph_repsenetation(x, edge_index, edge_type, batch)
        tlqw = 0
        tlpw = 0
        for layer_idx, layer in enumerate(self.bayes_linear_layers):
            if layer_idx == 0 or layer_idx == 1:
                features = layer(features)
            else:
                features, lqw, lpw = layer(features, sample)
                tlqw = tlqw + lqw
                tlpw = tlpw + lpw
            if layer_idx == len(self.bayes_linear_layers) - 1:
                pass
            else:
                features = self.dropout(F.relu(features))
        return features, tlqw, tlpw

    def sample_predict(self, batched_data, Nsamples):
        # Just copies type from x, initializes new vector

        if batched_data.edge_attr is None:
            edge_type = torch.zeros(batched_data.edge_index.shape[1], dtype=torch.long).to(
                batched_data.edge_index.device)
        else:
            edge_type = batched_data.edge_attr
        if batched_data.x is None:
            x = degree(batched_data.edge_index[0], batched_data.num_nodes).long()
        else:
            x = batched_data.x
        edge_index, batch = batched_data.edge_index, batched_data.batch

        predictions = torch.zeros(Nsamples, x.shape[0], self.output_dim).to(x)  # (Nsamples, batch_size, n_output)
        tlqw_vec = np.zeros(Nsamples)
        tlpw_vec = np.zeros(Nsamples)
        predictions = []
        for i in range(Nsamples):
            y, tlqw, tlpw = self.forward(batched_data, sample=True)
            # predictions[i] = y
            predictions.append(y)
            tlqw_vec[i] = tlqw
            tlpw_vec[i] = tlpw
        # print(predictions[0])
        predictions = torch.stack(predictions, dim=0)  # num_samples, batch_size, n_output
        # print(predictions.shape)
        return predictions, tlqw_vec, tlpw_vec



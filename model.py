import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import Linear
from torch_geometric.utils import negative_sampling
from utils import *


def orto(x, full=False):
    o = torch.mm(x.t(), x)
    o = o - torch.eye(*o.shape, device=o.device)
    n = torch.norm(o, "fro")
    return torch.pow(n, 2), n, None if not full else o


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.lin1 = Linear(in_channels=input_dim, out_channels=hidden_dim)
        self.lin2 = Linear(in_channels=hidden_dim, out_channels=output_dim)
        self.activation = nn.LeakyReLU()
        # self.norm = nn.BatchNorm1d(output_dim, affine=False)

    def forward(self, x, edge_index=None):
        x = self.lin1(x)
        x = self.activation(x)
        x = self.lin2(x)
        x = self.activation(x)
        # x = self.norm(x)
        return x


class EdgeDecoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, phi_psi):
        """
        phi and psi are source node and target node representations corresponding to an edge list,
        their dimensions should be the same. This method returns a prediction for the edge list
        """
        s = int(phi_psi.shape[1] / 2)
        e_hat = torch.sigmoid((phi_psi[:, :s] * phi_psi[:, s:]).sum(dim=1))
        return e_hat


class HeNCler(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_cl, s=16):
        '''
        recon_loss_fn 'mse_loss', 'cos_sim', 'jaccard'
        '''
        super(HeNCler, self).__init__()

        self.feature_space_dim = output_dim
        self.num_cl = num_cl
        self.s = s
        self.mlp1 = MLP(input_dim, hidden_dim, output_dim)
        self.mlp2 = MLP(input_dim, hidden_dim, output_dim)
        self.manifold_param = nn.Parameter(nn.init.orthogonal_(torch.Tensor(2 * (self.s), output_dim)))

        input_dec = 2 * output_dim
        hidden_dec = int((input_dec + input_dim) / 2)
        self.node_decoder = nn.Sequential(nn.Linear(in_features=input_dec, out_features=hidden_dec), nn.LeakyReLU(),
                                          nn.Linear(in_features=hidden_dec, out_features=input_dim))
        self.recon_loss_fn = nn.MSELoss()
        self.edge_decoder = EdgeDecoder()

    def forward(self, data):
        phi = self.mlp1(data.x)
        psi = self.mlp2(data.x)

        d_out_inv = 1 / (phi @ (psi.T @ torch.ones((psi.shape[0], 1))) + 1e-5).flatten()
        d_in_inv = 1 / ((torch.ones((1, phi.shape[0])) @ phi) @ psi.T + 1e-5).flatten()

        phi = phi - d_out_inv.view(1, -1) @ phi / d_out_inv.sum()
        psi = psi - d_in_inv.view(1, -1) @ psi / d_in_inv.sum()

        W, V = self.manifold_param.T[:, :self.s], self.manifold_param.T[:, self.s:]

        e = phi @ W
        r = psi @ V

        loss_dict = {}
        if self.training:
            # wKSVD loss
            pp_loss = (torch.linalg.norm(e, dim=1, ord=2) * d_out_inv).sum() + (
                    torch.linalg.norm(r, dim=1, ord=2) * d_in_inv).sum()
            # pp_loss = pp_loss + reg_loss
            pp_loss = pp_loss / data.x.shape[0]

            # Orthogonality loss
            orto_loss = orto(self.manifold_param.T, full=False)[0]
            loss_dict.update({'pp_loss': pp_loss, 'orto_loss': orto_loss})

            phi_hat, psi_hat = e @ W.T, r @ V.T
            ## Node reconstruction loss

            x_hat = self.node_decoder(torch.cat([phi_hat, psi_hat], dim=1))
            loss_dict.update({'node_rec_loss': self.recon_loss_fn(data.x, x_hat)})

            # Edge reconstruction loss
            num_pos = data.x.shape[0] * 2
            neg_sampling_ratio = 1
            num_neg = int(num_pos * neg_sampling_ratio)

            pos_edges = data.edge_index.T[np.random.choice(np.arange(data.num_edges), (num_pos,)), :]
            neg_edges = negative_sampling(data.edge_index, num_nodes=data.x.shape[0], num_neg_samples=num_neg).T

            edges = torch.cat([pos_edges, neg_edges], 0)
            edge_representations = torch.cat([phi_hat[edges[:, 0]], psi_hat[edges[:, 1]]], 1)

            edge_labels = torch.zeros((edges.shape[0],), device=edges.device)
            edge_labels[:num_pos] = 1

            edge_rec_loss = F.binary_cross_entropy(self.edge_decoder(edge_representations).squeeze(), edge_labels)

            loss_dict.update({'edge_rec_loss': edge_rec_loss})

        return torch.cat([e, r], dim=1), loss_dict

    def param_state(self):
        param_stiefel, param_other = [], []
        for name, param in self.named_parameters():
            if param.requires_grad and not 'manifold' in name:
                param_other.append(param)
            elif 'manifold' in name:
                param_stiefel.append(param)
        return param_stiefel, param_other

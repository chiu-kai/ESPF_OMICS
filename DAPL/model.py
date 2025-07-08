import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from DAPL.dataprocess import *
from torch_geometric import data as DATA
from torch_geometric.nn import GCNConv, GINConv, GATConv, ChebConv, GAE, global_mean_pool, global_max_pool

from typing import List

# class GraphEncoder(nn.Module):
#     def __init__(self, input_dim: int, output_dim: int, dropout=0.2):
#         super(GraphEncoder, self).__init__()
#         self.conv1 = GATConv(input_dim, input_dim, heads=10)
#         self.conv2 = GCNConv(input_dim*10, input_dim*10)
#         self.fc_g1 = torch.nn.Linear(input_dim*10, 256)
#         self.fc_g2 = torch.nn.Linear(256, output_dim)
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, data: DATA.data):
#         x, edge_index = data.x, data.edge_index
#         x = self.conv1(x, edge_index)
#         x = self.relu(x)
#         x = self.conv2(x, edge_index)
#         x = self.relu(x)
#         x = self.fc_g1(x)
#         x = self.dropout(x)
#         x = self.fc_g2(x)
#         return x



class VariationalGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VariationalGCNEncoder, self).__init__()
        # self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=True) # cached only for transductive learning
        # self.conv_mu = GCNConv(2 * out_channels, out_channels, cached=True)
        # self.conv_logstd = GCNConv(2 * out_channels, out_channels, cached=True)
        self.conv1 = ChebConv(in_channels, 2 * out_channels, K=2)
        self.bn1 = nn.BatchNorm1d(2 * out_channels)
        self.conv_mu = ChebConv(2 * out_channels, out_channels, K=2)
        self.conv_logstd = ChebConv(2 * out_channels, out_channels, K=2)
        self.relu = nn.ReLU()
    def forward(self, data: DATA.data, batch=None):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = self.relu(x)
        mu, logstd = self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)
        # mu = global_mean_pool(mu, batch=batch)
        # logstd = global_mean_pool(logstd, batch=batch)
        return mu, logstd


class gEncoder(nn.Module):
    def __init__(self, input_dim:int, output_dim:int, dropout=0.2):
        super(gEncoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim//2),
            nn.BatchNorm1d(input_dim//2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(input_dim//2, output_dim)
        )
    def forward(self, x):
        return self.net(x)


class gDecoder(nn.Module):
    def __init__(self, recon_dim:int, emb_dim:int, dropout=0.2):
        super(gDecoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, emb_dim*2),
            nn.BatchNorm1d(emb_dim*2),
            nn.ReLU(),
            # nn.Dropout(p=dropout),
            nn.Linear(emb_dim*2, recon_dim)
        )
    def forward(self, x):
        return self.net(x)


class GraphEncoder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout=0.2):
        super(GraphEncoder, self).__init__()
        # self.conv1 = GATConv(input_dim, input_dim, heads=10)
        # self.conv2 = GCNConv(input_dim*10, input_dim*10)
        # self.fc_g1 = torch.nn.Linear(input_dim*10, 256)
        # self.fc_g2 = torch.nn.Linear(256, output_dim)
        self.conv1 = ChebConv(in_channels=input_dim, out_channels=128, K=2)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = ChebConv(in_channels=128, out_channels=128, K=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc_g1 = torch.nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc_g2 = torch.nn.Linear(64, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, data: DATA.data, batch=None):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.fc_g1(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc_g2(x)
        x_mean = global_mean_pool(x, batch=batch)
        # x_mean = global_max_pool(x, batch=batch)
        return x, x_mean


class GraphDecoder(nn.Module):
    def __init__(self, recon_dim: int, emb_dim: int, dropout=0.2):
        super(GraphDecoder, self).__init__()
        self.fc_g1 = torch.nn.Linear(emb_dim, 1024)
        self.fc_g2 = torch.nn.Linear(1024, recon_dim)
        # self.conv1 = GCNConv(recon_dim, recon_dim)
        self.conv1 = ChebConv(recon_dim, recon_dim, K=2)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor, edge_index:torch.Tensor):
        x = self.fc_g1(x)
        x = self.relu(x)
        x = self.fc_g2(x)
        x = self.relu(x)
        x = self.conv1(x, edge_index)
        # x = self.relu(x)
        # x = self.conv1(x)
        return x


class Edgeindexdecoder(nn.Module):
    def __init__(self, input_dim:int):
        super(Edgeindexdecoder, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, input_dim)
        self.fc2 = torch.nn.Linear(input_dim, input_dim)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p=0.2)

    def forward(self, edge_index:torch.tensor):
        edge_index = self.fc1(edge_index)
        edge_index = self.relu(edge_index)
        edge_index = self.drop(edge_index)
        edge_index = self.fc2(edge_index)
        return edge_index

class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int] = None, dop: float = 0.1, act_fn=nn.SELU, out_fn=None, gr_flag=False, **kwargs) -> None:
        super(MLP, self).__init__()
        self.output_dim = output_dim
        self.dop = dop

        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        modules = []
        if gr_flag:
            modules.append(RevGrad())

        modules.append(
            nn.Sequential(
                nn.Linear(input_dim, hidden_dims[0], bias=True),
                # 若需要 BatchNorm，可自行解除以下註解
                # nn.BatchNorm1d(hidden_dims[0]),
                act_fn(),
                nn.Dropout(self.dop)
            )
        )
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i + 1], bias=True),
                    # nn.BatchNorm1d(hidden_dims[i + 1]),
                    act_fn(),
                    nn.Dropout(self.dop)
                )
            )
        self.module = nn.Sequential(*modules)
        if out_fn is None:
            self.output_layer = nn.Sequential(
                nn.Linear(hidden_dims[-1], output_dim, bias=True)
            )
        else:
            self.output_layer = nn.Sequential(
                nn.Linear(hidden_dims[-1], output_dim, bias=True),
                out_fn()
            )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        embed = self.module(input)
        output = self.output_layer(embed)
        return output

from torch.autograd import Function
class RevGrad(Function):
    @staticmethod
    def forward(ctx, input_, alpha_):
        ctx.save_for_backward(input_, alpha_)
        output = input_
        return output

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
        grad_input = None
        _, alpha_ = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = -grad_output * alpha_
        return grad_input, None


revgrad = RevGrad.apply


class RevGrad(torch.nn.Module):
    def __init__(self, alpha=1., *args, **kwargs):
        """
        A gradient reversal layer.
        This layer has no parameters, and simply reverses the gradient
        in the backward pass.
        """
        super().__init__(*args, **kwargs)

        self._alpha = torch.tensor(alpha, requires_grad=False)

    def forward(self, input_):
        return revgrad(input_, self._alpha)

class Classify(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int] = [10], dop: float = 0.1, act_fn=nn.ReLU, out_fn=None):
        super(Classify, self).__init__()
        # 這裡 hidden_dims=[10] 與 output_dim=1 將產生與原本相同的結構
        self.net = MLP(input_dim=input_dim, output_dim=1, hidden_dims=hidden_dims, dop=dop, act_fn=act_fn, out_fn=out_fn, gr_flag=False)
    
    def forward(self, x):
        # 輸出 reshape 為一維
        return self.net(x).view(-1)


class Classify_savefeature(nn.Module): 
    def __init__(self, input_dim: int, hidden_dims: List[int] = [10], dop: float = 0.1, act_fn=nn.ReLU, out_fn=None):
        super(Classify, self).__init__()
        # 這裡 hidden_dims=[10] 與 output_dim=1 將產生與原本相同的結構
        self.net = MLP(input_dim=input_dim, output_dim=1, hidden_dims=hidden_dims, dop=dop, act_fn=act_fn, out_fn=out_fn, gr_flag=False)

    def forward(self, x):
        return self.net(x), (self.net(x)).view(-1)
    

class projector(torch.nn.Module): #無用
    def __init__(self, in_dim, out_dim):
        super(projector, self).__init__()
        self.l1 = nn.Linear(in_dim, in_dim)
        self.bn = nn.BatchNorm1d(in_dim)
        self.re = nn.ReLU()
        self.l2 = nn.Linear(in_dim, out_dim)
    def forward(self, x):
        xout = self.l2(self.re(self.bn(self.l1(x))))
        return xout


class projector_decoder(torch.nn.Module): #無用
    def __init__(self, in_dim, out_dim):
        super(projector_decoder, self).__init__()
        self.l1 = nn.Linear(in_dim, in_dim)
        self.bn = nn.BatchNorm1d(in_dim)
        self.re = nn.ReLU()
        self.l2 = nn.Linear(in_dim, out_dim)
    def forward(self, x):
        xout = self.l2(self.re(self.bn(self.l1(x))))
        return xout

class Discriminator(nn.Module):
    def __init__(self, input_dim: int, dop: float = 0.0):
        super(Discriminator, self).__init__()
        # 使用 MLP 建立隱藏層結構：從 input_dim 映射到 input_dim//2，再映射到 1
        self.net = MLP(input_dim=input_dim, output_dim=1, hidden_dims=[input_dim // 2],
                       dop=dop, act_fn=nn.ReLU, out_fn=None, gr_flag=False)
    
    def forward(self, x):
        return self.net(x)

class relation_model(nn.Module):
    def __init__(self, indim, outdim=1):
        super(relation_model, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(indim, indim//2),
            nn.ReLU(),
            nn.Linear(indim//2, outdim),
            nn.Sigmoid()
        )
    def forward(self, x:torch.tensor):
        xout = (self.net(x)).view(-1)
        return xout


class VAE_Encoder(nn.Module):
    def __init__(self, input_size: int, latent_size: int, hidden_dims: List[int] = None, dop: float = 0.1, act_fn=nn.ReLU):
        super(VAE_Encoder, self).__init__()
        # 使用 MLP 將 input 映射到 hidden 表示
        # MLP 的輸出維度我們設定為 hidden_dims[-1]（若未給定則預設值可自行設定）
        hidden_dims = hidden_dims if hidden_dims is not None else [128, 64]
        self.mlp = MLP(input_dim=input_size, output_dim=hidden_dims[-1],
                       hidden_dims=hidden_dims, dop=dop, act_fn=act_fn)
        # 分別用兩個線性層產生 μ 與 σ
        self.mu_layer = nn.Linear(hidden_dims[-1], latent_size)
        self.sigma_layer = nn.Linear(hidden_dims[-1], latent_size)
        
    def forward(self, x: torch.Tensor):
        h = self.mlp(x)
        mu = self.mu_layer(h)
        sigma = self.sigma_layer(h)
        return mu, sigma

# 改寫的 VAE_Decoder 使用 MLP，並保留 hidden_dims 的彈性配置
class VAE_Decoder(nn.Module):
    def __init__(self, latent_size: int, output_size: int, hidden_dims: List[int] = None, dop: float = 0.1, act_fn=nn.ReLU):
        super(VAE_Decoder, self).__init__()
        # 使用 MLP 將 latent 向量映射到最終輸出
        # 這裡的 hidden_dims 決定中間隱藏層的結構
        hidden_dims = hidden_dims if hidden_dims is not None else [64, 128]
        self.mlp = MLP(input_dim=latent_size, output_dim=output_size,
                       hidden_dims=hidden_dims, dop=dop, act_fn=act_fn)
        
    def forward(self, x: torch.Tensor):
        return self.mlp(x)

# VAE 模型利用上述 Encoder 與 Decoder
class VAE(nn.Module):
    def __init__(self, input_size: int, output_size: int, latent_size: int, 
                 encoder_hidden_dims: List[int] = None, decoder_hidden_dims: List[int] = None,
                 dop: float = 0.1, act_fn=nn.ReLU):
        super(VAE, self).__init__()
        self.encoder = VAE_Encoder(input_size=input_size, latent_size=latent_size, 
                                   hidden_dims=encoder_hidden_dims, dop=dop, act_fn=act_fn)
        self.decoder = VAE_Decoder(latent_size=latent_size, output_size=output_size, 
                                   hidden_dims=decoder_hidden_dims, dop=dop, act_fn=act_fn)
        
    def forward(self, x: torch.Tensor):
        mu, sigma = self.encoder(x)
        std = torch.exp(0.5 * sigma)
        eps = torch.randn_like(std)
        z = mu + eps * sigma
        re_x = self.decoder(z)
        return re_x, z, mu, sigma

# 初始化權重函式
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    

class VAE_onedecoder(torch.nn.Module): #無用
    def __init__(self, input_size, decoder, latent_size, hidden_size):
        super(VAE_onedecoder, self).__init__()
        self.encoder = VAE_Encoder(input_size, hidden_size, latent_size)
        self.decoder = decoder
        # self.decoder = VAE_Decoder(latent_size, hidden_size, output_size)
    def forward(self, x): #x: bs,input_size
        mu,sigma = self.encoder(x) #mu,sigma: bs,latent_size
        std = torch.exp(0.5 * sigma)
        eps = torch.randn_like(std)  #eps: bs,latent_size
        z = mu + eps*sigma  #z: bs,latent_size
        re_x = self.decoder(z) # re_x: bs,output_size
        return re_x,z,mu,sigma

def vaeloss(mu, sigma, re_x, x, alpha=0.1):
        mseloss = torch.nn.MSELoss()
        recon_loss = mseloss(re_x, x)
        KLD = -0.5 * torch.sum(1 + sigma - mu.pow(2) - sigma.exp())
        loss = alpha*KLD+recon_loss
        return loss


import torch
import numpy as np
from torch import nn, einsum
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from .Dynamic_Static_Attention import Dynamic_Static_Attention

# ViT class
class myViT(nn.Module):

    def __init__(self, num_patches, dimensions, num_classes, dim, depth, heads, mlp_dim, pool = 'mean', channels = 1, dim_head = 32, dropout = 0., emb_dropout = 0.): # dim_head: dk, the dimension of matricers Q, K, and V.
        super().__init__()

        self.patch_dim_global = channels * dimensions

        # self.index = index

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding_global = nn.Sequential(
            # Rearrange('(b h) w -> b h w', b=1),
            nn.Linear(self.patch_dim_global, dim),
        )

        self.pos_embedding_global = nn.Parameter(torch.randn(1, num_patches + 1, dim)) 

        self.cls_token_global = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)


        self.transformer_global = Transformer_Global(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, num_classes)
            # nn.Softmax(dim=-1)
        )

        self.Layernorm_inter = nn.LayerNorm(dim)

        
    def forward(self, data):
        
        
        # =================Global Information
        #x = data
        x = self.to_patch_embedding_global(data)

        # x = x[:, np.argsort(self.index), :] # rearrange brain index
        b, n, _ = x.shape
        # Introduce patch 0
        cls_tokens_global = repeat(self.cls_token_global, '() n d -> b n d', b = b)

        x = torch.cat((cls_tokens_global, x), dim=1)

        # Positional embedding

        x += self.pos_embedding_global[:, :(n+1)]

        # =================================
        x_ori = self.dropout(x)

        # # ===============Global
        x_global = self.transformer_global(x_ori)
        
        
        x_global = self.Layernorm_inter(x_global)
        
        x_global_out = x_global.mean(dim=1)
#       
        out = self.to_latent(x_global_out)
        out = self.mlp_head(out)

        
        
        return out

      


#     ======================================Global==============================

class Transformer_Global(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm_Global(self.dim ,Dynamic_Static_Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm_Global(self.dim ,FeedForward_Global(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        # print(x.shape)
        # Internally mainly responsible for the forward part of the block.
        for attn, ff in self.layers:
            # print(x.shape)
            x= attn(x) + x
            # print(x.shape)
            x = ff(x) + x
        return x


# Layer norm
class PreNorm_Global(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)




# Fully connected layer, passing through one FC layer, then GELU, and then another FC layer
class FeedForward_Global(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)





if __name__ == "__main__":
    model = myViT(
        num_patches=216,
        # index=area_index,
        dimensions=93,
        num_classes=2,
        dim=512,  # 256,
        depth=2,  # 4,
        heads=2,  # 4,
        mlp_dim=512,  # 1024,
        pool='mean',
        # device = device,
        dropout=0.1,
        emb_dropout=0.1
    )
    data = np.random.randn(2,216,93)
    data = torch.from_numpy(data)
    # print("Input data type:", data.dtype)
    # print("Weight type in to_patch_embedding:", model.to_patch_embedding[1].weight.dtype)
    data = data.float()


    model(data)
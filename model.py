import torch.nn as nn
import torch as tc
import utils


import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
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

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class MapFormer(nn.Module):
    def __init__(self, args):
        super(MapFormer, self).__init__()
        d_model = args.d_model
        n_heads = args.n_heads
        n_layers = args.n_layers
        dropout_rate = args.dropout_rate
        layer_norm_eps = args.layer_norm_eps
        patch_size = 4

        self.position_embedding = utils.get_position_embedding().to(utils.get_device(args.use_cuda))
        
        self.patch_to_embedding = nn.Linear(patch_size * patch_size, d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, d_model, dropout_rate)
        encoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, n_layers, encoder_norm)

        decoder_layer = nn.TransformerDecoderLayer(d_model, n_heads, d_model, dropout_rate)
        decoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, n_layers, decoder_norm)

        self.output = nn.Linear(d_model, patch_size * patch_size)

        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        glimpses, locations, query_locations = inputs
        targets = utils.collapse_last_dim(self.position_embedding[query_locations], dim=3)
        locations = utils.collapse_last_dim(self.position_embedding[locations], dim=3)
        glimpses = self.patch_to_embedding(glimpses)
        source = glimpses + locations
        memory = self.transformer_encoder(source)
        output = self.transformer_decoder(targets, memory)
        output = self.output(output)
        return self.sigmoid(output)


class MapFormer_classifier(nn.Module):
    def __init__(self, args):
        super(MapFormer_classifier, self).__init__()
        d_model = args.d_model
        n_heads = args.n_heads
        n_layers = args.n_layers
        dropout_rate = args.dropout_rate
        layer_norm_eps = args.layer_norm_eps
        patch_size = 4

        self.cls_token = nn.Parameter(tc.randn(1, 1, d_model))
        self.cls_pos_token = nn.Parameter(tc.randn(1, 1, d_model))

        self.position_embedding = utils.get_position_embedding().to(utils.get_device(args.use_cuda))
        
        self.patch_to_embedding = nn.Linear(patch_size * patch_size, d_model)

        # encoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.transformer_encoder = Transformer(d_model, n_layers, n_heads, d_model, 128, dropout = dropout_rate)

        self.output =  nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs):
        glimpses, locations= inputs
        locations = utils.collapse_last_dim(self.position_embedding[locations], dim=3)
        batch_size = glimpses.shape[0]
        glimpses = self.patch_to_embedding(glimpses)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        glimpses = tc.cat((cls_tokens, glimpses), dim=1)
        cls_pos_tokens = self.cls_pos_token.expand(batch_size, -1, -1)
        locations = tc.cat((cls_pos_tokens, locations), dim=1)

        source = glimpses + locations
        encoding = self.transformer_encoder(source)
        output = self.output(encoding[:, 0])
        return self.softmax(output)
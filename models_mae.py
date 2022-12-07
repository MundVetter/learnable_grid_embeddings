# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

from timm.models.vision_transformer import PatchEmbed, Block

import utils.misc as misc
import utils.pos_embed as pos_embed

class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=partial(nn.LayerNorm, eps=1e-6), norm_pix_loss=False, pos_encoding = 'grid', encoding_type = 'hexagon',use_cuda=True, factor = 100, n_patches=10, **kwargs):
        super().__init__()
        assert img_size % patch_size == 0, 'img_size must be divisible by patch_size'
        assert embed_dim % num_heads == 0, 'embed_dim must be divisible by num_heads'
        assert decoder_embed_dim % decoder_num_heads == 0, 'decoder_embed_dim must be divisible by decoder_num_heads'
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = nn.Linear(patch_size * patch_size * in_chans, embed_dim)
        self.patch_size = patch_size
        self.in_chans = in_chans

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.keep_length = n_patches
        
        self.pos_encoding = pos_encoding
        if pos_encoding == 'grid':
            function = getattr(pos_embed, f'{encoding_type}_encoding')
            self.pos_embed = pos_embed.generate_grid_posembed(img_size, embed_dim, factor, encode_function=function).to(misc.get_device(use_cuda))
        else:
            self.pos_embed = misc.get_position_embedding(img_size, embed_dim, factor).to(misc.get_device(use_cuda))

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_pos_embed = self.compute_positional_encoding(misc.get_grid_locations(img_size, patch_size).unsqueeze(0)).to(misc.get_device(use_cuda))
        self.decoder_n_mask_tokens = len(self.decoder_pos_embed[0])

        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
         # fixed sin-cos embedding
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for _ in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], self.in_chans, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * self.in_chans))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_size
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, self.in_chans))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], self.in_chans, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore
    
    def compute_positional_encoding(self, locations):
        """
        locations: [N, L, 2], (x, y) of each patch

        TODO: add support for different embeddings for encoder and decoder
        """
        if self.pos_encoding == 'grid':
            x, y = locations[:,:, 0], locations[:,:, 1]
            return self.pos_embed[x, y]
        else:
            return misc.collapse_last_dim(self.pos_embed[locations], dim=3)

    def forward_encoder(self, x, pos_embed):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + pos_embed

        # append cls token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x

    def forward_decoder(self, x, pos_embed):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], self.decoder_n_mask_tokens, 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token

        # add pos embed
        x_ = x_ + torch.cat([pos_embed, self.decoder_pos_embed.repeat(x.shape[0], 1, 1)], dim=1)

        # append cls token
        x = torch.cat([x[:, :1, :], x_], dim=1)

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # keep only mask 
        x = x[:, 1 + self.keep_length:, :]

        return x

    def forward_loss(self, imgs, pred, patches):
        """
        imgs: [N + keep_length, 3, H, W]
        pred: [N, L, p*p*3]
        """
        # target = torch.cat([patches, self.patchify(imgs)], dim = 1)
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        # loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        # loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss.mean()

    def forward(self, patches, locations, imgs):
        pos_embed = self.compute_positional_encoding(locations)
        latent = self.forward_encoder(patches, pos_embed)
        pred = self.forward_decoder(latent, pos_embed)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred, patches)
        return loss, pred
        # return loss, pred[:, self.keep_length:, :], pred[:, :self.keep_length, :]


def mae_small_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=8, embed_dim=384, depth=12, num_heads=6,
        decoder_embed_dim=384, decoder_depth=8, decoder_num_heads=6,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


mae_vit_small_patch6 = mae_small_vit_base_patch16_dec512d8b
# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks
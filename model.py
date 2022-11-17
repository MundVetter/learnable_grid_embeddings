import torch.nn as nn
import utils


class MapFormer(nn.Module):
    def __init__(self, args):
        super(MapFormer, self).__init__()
        d_model = args.d_model
        n_heads = args.n_heads
        n_layers = args.n_layers
        dropout_rate = args.dropout_rate
        layer_norm_eps = args.layer_norm_eps

        self.position_embedding = utils.get_position_embedding()

        encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, d_model, dropout_rate)
        encoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, n_layers, encoder_norm)

        decoder_layer = nn.TransformerDecoderLayer(d_model, n_heads, d_model, dropout_rate)
        decoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, n_layers, decoder_norm)

        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        glimpses, locations, query_locations = inputs
        targets = utils.collapse_last_dim(self.position_embedding[query_locations], dim=3)
        locations = utils.collapse_last_dim(self.position_embedding[locations], dim=3)
        source = glimpses + locations
        memory = self.transformer_encoder(source)
        output = self.transformer_decoder(targets, memory)
        return self.sigmoid(output)

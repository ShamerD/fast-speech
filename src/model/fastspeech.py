import torch
from torch import nn

from src.model.default_config import ModelConfig
from src.model.fs_blocks import FFTBlock, LengthRegulator
from src.utils.data import Batch


class Encoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.emb = nn.Embedding(config.enc_vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.enc_pos_size, config.d_model)

        self.net = nn.Sequential(
            *[FFTBlock(config) for _ in range(config.n_encoder_layers)]
        )

    def forward(self, x):
        """
        :param x: tokens of shape [B, N]
        :return: encoder sequence of shape [B, N, d_model]
        """
        pos = torch.arange(x.size(1), device=x.device)
        pos = torch.minimum(pos, torch.full_like(pos, self.pos_emb.num_embeddings - 1))
        x_pos = self.pos_emb(pos.unsqueeze(0))

        x = self.emb(x)
        x = x + x_pos

        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.pos_emb = nn.Embedding(config.dec_pos_size, config.d_model)

        self.net = nn.Sequential(
            *[FFTBlock(config) for _ in range(config.n_decoder_layers)]
        )

    def forward(self, x):
        """
        :param x: extended sequence of shape [B, N_new, d_model]
        :return: new sequence of shape [B, N_new, d_model]
        """
        pos = torch.arange(x.size(1), device=x.device)
        pos = torch.minimum(pos, torch.full_like(pos, self.pos_emb.num_embeddings - 1))
        x_pos = self.pos_emb(pos.unsqueeze(0))

        x = x + x_pos

        return self.net(x)


class FastSpeech(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.encoder = Encoder(config)
        self.length_regulator = LengthRegulator(config)
        self.decoder = Decoder(config)
        self.mel_pred = nn.Linear(config.d_model, config.n_mels)

    def forward(self, batch: Batch):
        x = self.encoder(batch.tokens)
        x, log_pred_lens = self.length_regulator(x, batch.durations)
        x = self.decoder(x)
        mels = self.mel_pred(x).transpose(-1, -2)  # to make it [B, n_mels, N]

        return mels, log_pred_lens

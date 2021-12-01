from dataclasses import dataclass
from typing import Tuple, Optional, List

import torch
from torch.nn.utils.rnn import pad_sequence


@dataclass
class Batch:
    waveform: torch.Tensor                              # [B, L]
    waveform_length: torch.Tensor                       # [B]

    transcript: List[str]
    tokens: torch.Tensor                                # [B, N]
    token_lengths: torch.Tensor                         # [B]

    durations: Optional[torch.Tensor] = None            # [B, N]
    durations_pred: Optional[torch.Tensor] = None       # [B, N]

    mels: Optional[torch.Tensor] = None                 # [B, n_mels, N]
    mels_pred: Optional[torch.Tensor] = None            # [B, n_mels, N']

    mel_loss: Optional[torch.Tensor] = None
    dur_loss: Optional[torch.Tensor] = None
    loss: Optional[torch.Tensor] = None

    def to(self, device: torch.device) -> 'Batch':
        self.waveform = self.waveform.to(device)
        self.waveform_length = self.waveform_length.to(device)

        self.tokens = self.tokens.to(device)
        self.token_lengths = self.token_lengths.to(device)

        if self.durations is not None:
            self.durations = self.durations.to(device)
        if self.durations_pred is not None:
            self.durations_pred = self.durations_pred.to(device)

        if self.mels is not None:
            self.mels = self.mels.to(device)
        if self.mels_pred is not None:
            self.mels_pred = self.mels_pred.to(device)

        return self


class LJSpeechCollator:

    def __call__(self, instances: List[Tuple]) -> Batch:
        waveform, waveform_length, transcript, tokens, token_lengths = list(
            zip(*instances)
        )

        waveform = pad_sequence([
            waveform_[0] for waveform_ in waveform
        ]).transpose(0, 1)
        waveform_length = torch.cat(waveform_length)

        tokens = pad_sequence([
            tokens_[0] for tokens_ in tokens
        ]).transpose(0, 1)
        token_lengths = torch.cat(token_lengths)

        return Batch(waveform, waveform_length, transcript, tokens, token_lengths)

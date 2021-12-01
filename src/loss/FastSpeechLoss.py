from torch import nn

from src.utils.data import Batch


class FastSpeechLoss(nn.Module):
    def __init__(
            self,
            mel_loss=nn.MSELoss(),
            dur_loss=nn.MSELoss(),
    ):
        super().__init__()
        self.mel_loss = mel_loss
        self.dur_loss = dur_loss

    def forward(self, batch: Batch):
        assert batch.durations.size() == batch.durations_pred.size()

        dur_loss = self.dur_loss(batch.durations, batch.durations_pred)

        min_seq_len = min(batch.mels.size(-1), batch.mels_pred.size(-1))
        mel_loss = self.mel_loss(batch.mels[:, :, :min_seq_len], batch.mels_pred[:, :, :min_seq_len])

        return mel_loss, dur_loss

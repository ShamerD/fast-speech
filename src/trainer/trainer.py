import random
from random import shuffle
from typing import Optional, List

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
import torchaudio

import PIL
from tqdm import tqdm

from .base import BaseTrainer
from src.utils.config_parser import ConfigParser
from src.utils.data import Batch, MelSpectrogram
from src.model import FastSpeech, Waveglow, GraphemeAligner
from src.logger.utils import plot_spectrogram_to_buf
from src.utils import inf_loop, MetricTracker


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
            self,
            model: FastSpeech,
            wav2mel: MelSpectrogram,
            aligner: GraphemeAligner,
            vocoder: Waveglow,
            criterion: nn.Module,
            metrics: List,
            optimizer: torch.optim.Optimizer,
            config: ConfigParser,
            device: torch.device,
            data_loader: torch.utils.data.DataLoader,
            valid_data_loader=None,
            lr_scheduler=None,
            len_epoch: Optional[int] = None,
            skip_oom: bool = True,
    ):
        super().__init__(model, criterion, metrics, optimizer, config, device)
        self.wav2mel = wav2mel
        self.aligner = aligner
        self.vocoder = vocoder

        self.skip_oom = skip_oom
        self.config = config
        self.data_loader = data_loader

        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch

        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None

        self.lr_scheduler = lr_scheduler
        self.log_step = 100

        self.train_metrics = MetricTracker(
            "loss", "mel_loss", "duration_loss", "grad norm",
            *[m.name for m in self.metrics if m.use_on_train], writer=self.writer
        )
        self.valid_metrics = MetricTracker(
            "loss", "mel_loss", "duration_loss",
            *[m.name for m in self.metrics if m.use_on_val], writer=self.writer
        )

    def _clip_grad_norm(self):
        if self.config["trainer"].get("grad_norm_clip", None) is not None:
            clip_grad_norm_(
                self.model.parameters(), self.config["trainer"]["grad_norm_clip"]
            )

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        self.writer.add_scalar("epoch", epoch)

        for batch_idx, batch in enumerate(
                tqdm(self.data_loader, desc="train", total=self.len_epoch)
        ):
            if batch_idx >= self.len_epoch:
                break

            try:
                batch = self.process_batch(
                    batch,
                    is_train=True,
                    metrics=self.train_metrics,
                )
            except RuntimeError as e:
                if "out of memory" in str(e) and self.skip_oom:
                    self.logger.warning("OOM on batch. Skipping batch.")
                    for p in self.model.parameters():
                        if p.grad is not None:
                            del p.grad  # free some memory
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e

            self.train_metrics.update("grad norm", self.get_grad_norm())
            if batch_idx % self.log_step == 0:
                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                self.logger.debug(
                    "Train Epoch: {} {} Loss: {:.6f}".format(
                        epoch, self._progress(batch_idx), batch.loss.item()
                    )
                )
                self.writer.add_scalar(
                    "learning rate", self.lr_scheduler.get_last_lr()[0]
                )
                self._log_predictions(batch)
                self._log_scalars(self.train_metrics)

        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{"val_" + k: v for k, v in val_log.items()})

        return log

    def process_batch(
            self,
            batch: Batch,
            is_train: bool,
            metrics: MetricTracker
    ):
        batch = batch.to(self.device)

        with torch.no_grad():
            batch.mels = self.wav2mel(batch.waveform)

            # TODO: what to do with validation?
            batch.durations = self.aligner(batch.waveform, batch.waveform_length, batch.transcript) * batch.mels.size(-1)

        if is_train:
            self.optimizer.zero_grad()

        mels, log_lengths = self.model(batch)

        batch.mels_pred = mels
        batch.mel_loss, batch.dur_loss = self.criterion(batch)
        batch.loss = batch.mel_loss + batch.dur_loss

        if is_train:
            batch.loss.backward()
            self._clip_grad_norm()
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        metrics.update("loss", batch.loss.item())
        metrics.update("mel_loss", batch.mel_loss.item())
        metrics.update("duration_loss", batch.dur_loss.item())

        for met in self.metrics:
            if met.name in metrics.keys():
                metrics.update(met.name, met(batch))

        return batch

    @torch.no_grad()
    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch
        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()

        batch = None
        for batch_idx, batch in tqdm(
                enumerate(self.valid_data_loader),
                desc="validation",
                total=len(self.valid_data_loader),
        ):
            batch = self.process_batch(
                batch,
                is_train=False,
                metrics=self.valid_metrics,
            )

        self.writer.set_step(epoch * self.len_epoch, "val")
        self._log_scalars(self.valid_metrics)
        self._log_predictions(batch)

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins="auto")
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.data_loader, "n_samples"):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    @torch.no_grad()
    def _log_predictions(
            self,
            batch: Batch,
    ):
        if self.writer is None:
            return

        idx = random.randrange(len(batch.transcript))

        self.writer.add_text("transcript", batch.transcript[idx])
        self._log_spectrogram("true spectrogram", batch.mels[idx])
        self._log_spectrogram("predicted spectrogram", batch.mels_pred[idx])
        self._log_audio("true audio", batch.waveform[idx, :batch.waveform_length[idx]])
        self._log_audio("generated audio", self.vocoder.inference(batch.mels_pred[idx]))

    def _log_spectrogram(self, spec_name, spectrogram):
        image = PIL.Image.open(plot_spectrogram_to_buf(spectrogram.cpu().log()))
        self.writer.add_image(spec_name, image)

    def _log_audio(self, audio_name, audio):
        self.writer.add_audio(audio_name, audio, self.wav2mel.config.sr)

    @torch.no_grad()
    def get_grad_norm(self, norm_type=2):
        parameters = self.model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).cpu() for p in parameters]
            ),
            norm_type,
        )
        return total_norm.item()

    def _log_scalars(self, metric_tracker: MetricTracker):
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(f"{metric_name}", metric_tracker.avg(metric_name))

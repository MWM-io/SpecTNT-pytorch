import mir_eval
import torch as th
from .base import BaseModel

class BeatEstimator(BaseModel):
    def __init__(self, feature_extractor, net, optimizer, lr_scheduler, criterion, datamodule, activation_fn):
        super().__init__(
            feature_extractor,
            net,
            optimizer,
            lr_scheduler,
            criterion,
            datamodule,
            activation_fn
        )
        
        self.target_fps = datamodule.sample_rate / \
            (datamodule.hop_length * datamodule.time_shrinking)

    def training_step(self, batch, batch_idx):
        losses = {}
        x, y = batch['audio'], batch['targets']
        features = self.feature_extractor(x)
        logits = self.net(features)
        losses['train_loss'] = self.criterion(
            logits.flatten(end_dim=1), y.flatten(end_dim=1))
        self.log_dict(losses, on_step=False, on_epoch=True)
        return losses['train_loss']

    def validation_step(self, batch, batch_idx):
        losses = {}
        audio, targets, ref_beats, ref_downbeats = (
            batch['audio'][0], 
            batch['targets'][0].cpu(), 
            batch['beats'][0].cpu(), 
            batch['downbeats'][0].cpu()
        )
        input_length, sample_rate, batch_size = (
            self.datamodule.input_length,
            self.datamodule.sample_rate,
            self.datamodule.batch_size
        )
        audio_chunks = th.cat([el.unsqueeze(0) for el in audio.split(
            split_size=int(input_length*sample_rate))[:-1]], dim=0)
        # Inference loop
        logits_list, probs_list = th.tensor([]), th.tensor([])
        for batch_audio in audio_chunks.split(batch_size):
            with th.no_grad():
                features = self.feature_extractor(batch_audio)
                logits = self.net(features)
                probs = self.activation(logits)
                logits_list = th.cat(
                    [logits_list, logits.flatten(end_dim=1).cpu()], dim=0)
                probs_list = th.cat(
                    [probs_list, probs.flatten(end_dim=1).cpu()], dim=0)
        # Postprocessing
        beats_data = probs_list.argmax(dim=1)
        est_beats = th.where(beats_data == 0)[0] / self.target_fps
        est_downbeats = th.where(beats_data == 1)[0] / self.target_fps
        # Eval
        losses['val_loss'] = self.criterion(
            logits_list, targets[:len(logits_list)])
        losses['beats_f_measure'] = mir_eval.beat.f_measure(
            ref_beats, est_beats)
        losses['downbeats_f_measure'] = mir_eval.beat.f_measure(
            ref_downbeats, est_downbeats)
        self.log_dict(losses, on_step=False, on_epoch=True)
        return losses['val_loss']

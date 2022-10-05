import torch as th
from sklearn import metrics
from .base import BaseModel


class MusicTagger(BaseModel):
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

    def training_step(self, batch, batch_idx):
        loss_dict = {}
        x, y = batch['audio'], batch['targets']
        features = self.feature_extractor(x)
        logits = self.net(features)
        loss_dict['train_loss'] = self.criterion(logits, y)
        self.log_dict(loss_dict, on_step=False, on_epoch=True)
        return loss_dict['train_loss']

    def validation_step(self, batch, batch_idx):
        loss_dict = {}
        x, y = batch['audio'][0], batch['targets'].cpu()
        sample_rate, input_length, batch_size = (
            self.datamodule.sample_rate,
            self.datamodule.input_length,
            self.datamodule.batch_size
        )
        # Process whole track as batches of chunks
        audio_chunks = th.cat([el.unsqueeze(0) for el in x.split(
            split_size=int(input_length*sample_rate))[:-1]], dim=0)
        logits_list, probs_list = th.tensor([]), th.tensor([])
        for audio_batch in audio_chunks.split(batch_size):
            with th.no_grad():
                features = self.feature_extractor(audio_batch)
                logits = self.net(features)
                probs = self.activation(logits)
                logits_list = th.cat([logits_list, logits.cpu()], dim=0)
                probs_list = th.cat([probs_list, probs.cpu()], dim=0)
        # Aggregate along track and then compute metrics
        logits_agg, probs_agg = logits_list.mean(dim=0).unsqueeze(
            0), probs_list.mean(dim=0).unsqueeze(0)
        loss_dict['val_loss'] = self.criterion(logits_agg, y).item()
        loss_dict['val_roc_auc'] = metrics.roc_auc_score(
            y.T, probs_agg.T, average="macro")
        loss_dict['val_pr_auc'] = metrics.average_precision_score(
            y.T, probs_agg.T, average="macro")
        self.log_dict(loss_dict, on_step=False, on_epoch=True)
        return loss_dict['val_loss']

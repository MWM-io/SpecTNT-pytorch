import torch as th
import hydra.utils as hu
from omegaconf import OmegaConf

def load_modules(config_path):
    cfg = OmegaConf.load(config_path):
    datamodule = hu.instantiate(cfg.datamodule)
    _ = datamodule.setup(None)
    feature_extractor = hu.instantiate(cfg.features)
    fe_model = hu.instantiate(cfg.fe_model)
    net = hu.instantiate(cfg.net, fe_model=fe_model)
    return datamodule, feature_extractor, net


def inference(audio, datamodule, feature_extractor, net, activation_fn):
    input_length, sample_rate, batch_size = (
        datamodule.input_length,
        datamodule.sample_rate,
        datamodule.batch_size
    )
    audio_chunks = th.cat([el.unsqueeze(0) for el in audio.split(
        split_size=int(input_length*sample_rate))[:-1]], dim=0)
    # Inference loop
    probs_list = th.tensor([])
    for batch_audio in audio_chunks.split(batch_size):
        with th.no_grad():
            features = feature_extractor(batch_audio)
            logits = net(features)
            if activation_fn == "softmax":
                probs = th.softmax(logits, dim=2)
            elif activation_fn == "sigmoid":
                probs = th.sigmoid(logits)
            probs_list = th.cat(
                [probs_list, probs.flatten(end_dim=1).cpu()], dim=0)
    return probs_list

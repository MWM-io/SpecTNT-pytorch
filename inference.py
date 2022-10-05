import librosa
import torch as th
import torch.nn as nn
import hydra.utils as hu
from omegaconf import OmegaConf


def predict(audio_path: str, cfg_path: str, ckpt_path: str) -> th.Tensor:
    """
    Args:
        audio_path: string path to audio file to be analyzed
        cfg_path: string path to config
        ckpt_path: string path to checkpoint

    Return:
        probs_list: torch.Tensor of estimated probability distribution over output classes for each output frame
    """
    # Load config and params
    cfg = OmegaConf.load(cfg_path)
    input_length, sample_rate, batch_size = (
        cfg.datamodule.input_length,
        cfg.datamodule.sample_rate,
        cfg.datamodule.batch_size
    )
    # Load audio
    audio, _ = librosa.load(audio_path, sr=sample_rate, mono=True)
    audio = th.from_numpy(audio)
    # Load modules
    feature_extractor = hu.instantiate(cfg.features)
    fe_model = hu.instantiate(cfg.fe_model)
    net = hu.instantiate(cfg.net, fe_model=fe_model)
    # Load weights
    if ckpt_path is not None:
        ckpt = th.load(ckpt_path, map_location="cpu")
        net_state_dict = {k.replace("net.", ""): v for k,
                          v in ckpt["state_dict"].items() if "feature_extractor" not in k}
        net.load_state_dict(net_state_dict)
        features_state_dict = {k.replace("feature_extractor.", ""): v for k,
                               v in ckpt["state_dict"].items() if "feature_extractor" in k}
        feature_extractor.load_state_dict(features_state_dict)
    _ = net.eval()
    _ = feature_extractor.eval()
    # Inference loop
    audio_chunks = th.cat([el.unsqueeze(0) for el in audio.split(
        split_size=int(input_length*sample_rate))[:-1]], dim=0)
    probs_list = th.tensor([])
    for batch_audio in audio_chunks.split(batch_size):
        with th.no_grad():
            features = feature_extractor(batch_audio)
            logits = net(features)
            if cfg.model.activation_fn == "softmax":
                probs = th.softmax(logits, dim=2)
            elif cfg.model.activation_fn == "sigmoid":
                probs = th.sigmoid(logits)
            probs_list = th.cat(
                [probs_list, probs.flatten(end_dim=1).cpu()], dim=0)
    return probs_list


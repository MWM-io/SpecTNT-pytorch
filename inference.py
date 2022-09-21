import torch as th
import hydra.utils as hu
from omegaconf import OmegaConf

def predict(audio, cfg_path, ckpt_path, activation_fn):
    """
    Args:
        audio: waveform as a 1D Pytorch tensor (sample rate: 16kHz for beat estimation and 22kHz for music tagging)
        cfg_path: string indicating config path
        ckpt_path: string indicating checkpoint path
        activation_fn: activation function, either "softmax" (beat estimation) or "sigmoid" (music tagging)
    
    Return:
        probs_list: list of estimated probability distribution over output classes for each output frame
    """
    # Load config and params
    cfg = OmegaConf.load(cfg_path)
    input_length, sample_rate, batch_size = (
        cfg.datamodule.input_length,
        cfg.datamodule.sample_rate,
        cfg.datamodule.batch_size
    )
    # Load modules
    feature_extractor = hu.instantiate(cfg.features)
    fe_model = hu.instantiate(cfg.fe_model)
    net = hu.instantiate(cfg.net, fe_model=fe_model)
    # Load weights
    ckpt = th.load(ckpt_path, map_location="cpu")
    net_state_dict = {k.replace("net.", ""): v for k,
                    v in ckpt["state_dict"].items() if "feature_extractor" not in k}
    net.load_state_dict(net_state_dict)
    _ = net.eval()
    features_state_dict = {k.replace("feature_extractor.", ""): v for k,
                        v in ckpt["state_dict"].items() if "feature_extractor" in k}
    feature_extractor.load_state_dict(features_state_dict)
    _ = feature_extractor.eval()
    # Inference loop
    audio_chunks = th.cat([el.unsqueeze(0) for el in audio.split(
        split_size=int(input_length*sample_rate))[:-1]], dim=0)
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

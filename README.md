# SpecTNT

Repository exploring the SpecTNT architecture applied to:
- music tagging as described in the paper ["SpecTNT: a time-frequency transformer for music audio"](https://arxiv.org/abs/2110.09127)
- beats and downbeats estimation as described in the paper ["Modeling beats and downbeats with a time-frequency transformer"](https://arxiv.org/abs/2205.14701)

### Disclaimer

This is by no mean an official implementation. Any comments and suggestions regarding fixes and enhancements are highly encouraged. 

## Datasets

Datasets constitution and their associated class implementation are excluded from this repo. Dummy classes are presented instead.

List of datasets used in paper associated to each task for training, validation and testing:
- music tagging: a Million Song Dataset subset split [this way](https://github.com/minzwon/semi-supervised-music-tagging-transformer/blob/master/data/splits/msd_splits.tsv) and intersected with the [LastFM dataset](http://millionsongdataset.com/lastfm/) for tags
- beats and downbeats estimation: Beatles, Ballroom, SMC, Hainsworth, Simac, HJDB, RWC-Popular, Harmonix Set

## Configuration

### Datamodule
- `input_length`: models are trained on audio chunks which length is defined here in seconds. 
- `hop_length`: number of samples between successive frames used to constructs features (melspectrograms or harmonic filters).
- `time_shrinking` *(specific to beat estimation)*: frequency and time poolings entail dimension shrinking along the time and the frequency axis after inference through the front-end model. Target tensors should be constructed taking this shrinking into account along the time axis. 

### Front-end model
- `freq_pooling`, `time_pooling` *(specific to beat estimation)*: pooling along the frequency and the time axis that occur at the end of the front-end module. As a crutial impact on training and inference times but are not clearly specified in the paper in our understanding.

### Network
- `n_channels`, `n_frequencies`, `n_times`: shape of tensors that input the SpecTNT. Should be consistent with datamodule and features parameters.
- `use_tct`: whether to use Temporal Class Tokens, which act as aggregators along the temporal axis, or not. Set to `true` in case of a track-wise prediction (ex: tagging) task and to `false` in case of a frame-wise prediction (ex: beat estimation) task. 
- `n_classes`: number of output classes. 

## Usage

### Inference
Inference functions are presented in `inference.py` and should guide users unfamiliar with `pytorch-lightning` and `hydra` libraries.  
Here is a quick example usage of those functions:
```python
from inference import load_modules, inference

# Load modules
datamodule, feature_extractor, net = load_modules(
    config_path="configs/beats.yaml"
)
# Load and prepare data
batch = next(iter(datamodule.val_dataloader()))
audio = batch['audio'][0]
# Inference
probs_list = inference(
    audio=audio,
    datamodule=datamodule,
    feature_extractor=feature_extractor,
    net=net,
    activation_fn="softmax"
)
```

### Training
:warning: Real training should be preceded with a dataset constitution following guidelines presented in each paper. 

Users can though test training pipelines with the dummy dataset classes presented in this repo, selecting a `config_name`, replacing it in `train.py` main function decorator and launching this command in the terminal:
```bash
python train.py
```

# SpecTNT

Repository exploring the SpecTNT architecture applied to:
- music tagging as described in the paper ["SpecTNT: a time-frequency transformer for music audio"](https://arxiv.org/abs/2110.09127)
- beats and downbeats estimation as described in the paper ["Modeling beats and downbeats with a time-frequency transformer"](https://arxiv.org/abs/2205.14701)

### Disclaimer

This is by no mean an official implementation. Any comments and suggestions regarding fixes and enhancements are highly encouraged. 

## Datasets

Datasets constitution and their associated class implementation are excluded from this repo. Dummy classes are presented instead.

:warning: as is, the implentation of validation steps in [models](/models/) entails that validation datamodules should stack entire tracks by batch of 1.

List of datasets used in the paper associated to each task for training, validation and testing:
- music tagging: a Million Song Dataset subset split [this way](https://github.com/minzwon/semi-supervised-music-tagging-transformer/blob/master/data/splits/msd_splits.tsv) and intersected with the [LastFM dataset](http://millionsongdataset.com/lastfm/) for tags
- beats and downbeats estimation: Beatles, Ballroom, SMC, Hainsworth, Simac, HJDB, RWC-Popular, Harmonix Set

## Configuration

### Datamodule
- `input_length`: models are trained on audio chunks which length is defined here in seconds. 
- `hop_length`: number of samples between successive frames used to constructs features (melspectrograms or harmonic filters).
- `time_shrinking` *(specific to beat estimation)*: time pooling entails dimension shrinking along the time axis after inference through the front-end model (see below for more details). Target labels tensors should be constructed taking this shrinking into account. Has a crutial impact on inference durations but is not clearly specified in the paper.

### Front-end model
- `freq_pooling`, `time_pooling` *(specific to beat estimation)*: pooling along the frequency and the time axis that occur at the end of the front-end module. Have a crutial impact on inference durations but are not clearly specified in the paper.

### Network
- `n_channels`, `n_frequencies`, `n_times`: shape of tensors that input the SpecTNT. Should be consistent with audio chunk shape and feature extractor parameters.
- `use_tct`: whether to use Temporal Class Tokens, which act as aggregators along the temporal axis, or not. Set to `true` in case of a track-wise prediction task (ex: tagging) and to `false` in case of a frame-wise prediction task (ex: beat estimation).
- `n_classes`: number of output classes. 

## Usage

### Inference
Inference functions are presented in `inference.py` and should guide users unfamiliar with `pytorch-lightning` and `hydra` libraries.  

### Training
:warning: Real training should be preceded with a dataset constitution following guidelines presented in each paper. 

Users can though test training pipelines with the dummy dataset classes presented in this repo using the following commands in terminal:
- beats and downbeats estimation:
```bash
python train.py --config-name beats
```
- music tagging:
```bash
python train.py --config-name tagging
```

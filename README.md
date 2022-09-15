# SpecTNT

Repository exploring the SpecTNT architecture applied to:
- music tagging as described in the paper ["SpecTNT: a time-frequency transformer for music audio"](https://arxiv.org/abs/2110.09127)
- beats and downbeats estimation as described in the paper ["Modeling beats and downbeats with a time-frequency transformer"](https://arxiv.org/abs/2205.14701)

## Disclaimer

This is by no mean an official implementation. Any comments and suggestions regarding fixes and enhancements are highly encouraged. 

## Datasets

Datasets constitution and their associated class implementation are excluded from this repo. Dummy classes are presented instead.

List of datasets used in each paper for training, validation and testing:
- music tagging: a Million Song Dataset subset split [this way](https://github.com/minzwon/semi-supervised-music-tagging-transformer/blob/master/data/splits/msd_splits.tsv) and intersected with the [LastFM dataset](http://millionsongdataset.com/lastfm/) for tags
- beats and downbeats estimation: Beatles, Ballroom, SMC, Hainsworth, Simac, HJDB, RWC-Popular, Harmonix Set

## Usage

Chose a task and constitute a dataset. Then replace `config_name` in `train.py` whith the chosen task and run `python train.py` in terminal.
Self-Supervised VQ-VAE for Zero-Shot Music Style Transfer
========================================================

This is the code repository for the *Zero-shot music timbre transfer by transfer learning from timbre verification* master thesis
and is mostly based on the [code repository](https://github.com/cifkao/ss-vq-vae) for the paper
*Self-Supervised VQ-VAE for One-Shot Music Style Transfer*
by Ondřej Cífka, Alexey Ozerov, Umut Şimşekli, and Gaël Richard.


Other repositories
------------------

The code for the thesis is contained within different repositories:

- [`random-midi-generator`](https://github.com/wojtekk23/random-midi-generator) - Fork of `xanderlewis`'s [`random-midi-generator`](https://github.com/xanderlewis/random-midi-generator)
- [`MIDIInstrumentSwap`](https://github.com/wojtekk23/MIDIInstrumentSwap) - Generating MIDI files of the same melody with many different instruments
- [`COLA-PyTorch`](https://github.com/wojtekk23/COLA-PyTorch) - Code for the PyTorch implementation of the [COLA](https://arxiv.org/abs/2010.10915) model

Models, outputs, datasets
-------------------------

- [Model outputs](https://drive.google.com/drive/folders/1VdM5QtXtb7ZUBPPnHEj3tRyP2NQBaZ2N?usp=sharing)
- [Synthesized MIDI Dataset](https://drive.google.com/drive/folders/1xDLJwC2hBSEuPHZwE634xvi8JZYL9RsB?usp=sharing)
- [Model weights, configurations and training histories](https://drive.google.com/drive/folders/1h64rJerW8LmXkDkKNK7WigqxRD4K7rXC?usp=sharing)

Contents
--------

- `src` – the main codebase (the `ss-vq-vae` package); install with `pip install ./src`; usage details [below](#Usage)
- `data` – Jupyter notebooks for data preparation (details [below](#Datasets))
- `experiments` – model configuration, evaluation, code for generating diagrams/visualizations, notebooks with analyses

Setup
-----

```sh
pip install -r requirements-frozen.txt
pip install -e ./src
```

Usage
-----

To train the model, go to `experiments`, then run:
```sh
python -m ss_vq_vae.models.vqvae_oneshot --logdir=model train
```
This is assuming the training data is prepared (see [below](#Datasets)).

You can also use the scripts provided in the top level directory of the repository:

- `train_finetuned_*.sh` - trains a COLA-based variant
- `train_original_*.sh` - trains a variant based on the original RNN style encoder

Datasets
--------
Each dataset used in the paper has a corresponding directory in `data`, containing a Jupyter notebook called `prepare.ipynb` for preparing the dataset:
- the entire training and validation dataset: `data/comb`; combined from LMD and RT (see below)
- [Lakh MIDI Dataset](https://colinraffel.com/projects/lmd/) (LMD), rendered as audio using SoundFonts
  - the part used as training and validation data: `data/lmd/audio_train`
  - the part used as the 'artificial' test set: `data/lmd/audio_test`
  - both require [downloading](http://hog.ee.columbia.edu/craffel/lmd/lmd_full.tar.gz) the raw data and pre-processing it using `data/lmd/note_seq/prepare.ipynb`
  - the following SoundFonts are required (available [here](https://packages.debian.org/buster/fluid-soundfont-gm) and [here](https://musescore.org/en/handbook/soundfonts-and-sfz-files#list)): `FluidR3_GM.sf2`, `TimGM6mb.sf2`, `Arachno SoundFont - Version 1.0.sf2`, `Timbres Of Heaven (XGM) 3.94.sf2`
- RealTracks (RT) from [Band-in-a-Box](https://www.pgmusic.com/) UltraPAK 2018 (not freely available): `data/rt`
- [Mixing Secrets](https://www.cambridge-mt.com/ms/mtk/) data
  - the 'real' test set: `data/mixing_secrets/test`
  - the set of triplets for training the timbre metric: `data/mixing_secrets/metric_train`
  - both require downloading and pre-processing the data using `data/mixing_secrets/download.ipynb`

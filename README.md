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

Models in the folders above:

- `run-without-violin-bowed-27-10-2023` - COLA-style style encoder
- `run-contrastive-original-without-violin-bowed-21-11-2023` - RNN-style style encoder
- `run-contrastive-original-style-metric-nsynth-12-07-2024` - RNN-style model used as the timbre similarity metric
- `model-original-no-style-pretraining-with-ssl-dataloader-07-09-2024` - Timbre transfer model with RNN-style style encoder, no style pretraining, using the SSL dataloader (like Cifka et al. 2021)
- `model-original-no-style-pretraining-19-11-2023` - Timbre transfer model with RNN-style style encoder, no style pretraining
- `model-original-frozen-style-pretraining-21-11-2023` - Timbre transfer model with RNN-style style encoder, style encoder frozen
- `model-original-finetuned-style-pretraining-22-11-2023` - Timbre transfer model with RNN-style style encoder, style encoder finetuned
- `model-leaky-relu-no-style-pretraining-30-08-2024` - Timbre transfer model with COLA-style style encoder, no style pretraining
- `model-leaky-relu-no-style-pretraining-13-11-2023` - Old/unused version of the above
- `model-leaky-relu-frozen-style-pretraining-01-09-2024` - Timbre transfer model with COLA-style style encoder, style encoder frozen
- `model-leaky-relu-frozen-style-pretraining-15-11-2023` - Old/unused version of the above
- `model-leaky-relu-finetuned-style-pretraining-29-08-2024` - Timbre transfer model with COLA-style style encoder, style encoder finetuned
- `model-leaky-relu-finetuned-style-pretraining-15-11-2023` - Old/unused version of the above


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

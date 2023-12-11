# Earth Observation Compression via Vector Quantization in PyTorch

## Description

The repository contains PyTorch implementations of VQ-(V)AE
(the "V" is dropped because we model the compressor as
a _non-variational_ autoencoder)
to use on earth observation data from the BigEarthNet dataset.
This dataset has been introduced as a benchmark
for multi-label classification (although the labels are irrelevant here)
on earth observations, in the context of remote sensing.
The dataset is a collection of currated samples
originating from the Sentinel-2 data source.
The goal of the model consists in learning to compress earth observations
in as few bits as possible.

There is one script to run the job locally (`anton_launcher`)
and one adapted for Slurm-orchestrated HPC clusters (`slurm_launcher`).
The file `spawner.py` at the root (called by `slurm_launcher`)
also offer the option to spawn arrays of jobs locally within a Tmux session
(where each job spawned has its own window within the session).

When run locally, the scripts expect the dataset to be present locally _uncompressed_.
When run with Slurm, it expects the dataset to be present on a accessible note _compressed_.
These behaviors are modifiable in the scripts provided.
The choices I made here were for my own convenience.

## Requirements

GDAL: might need to install `gdal` and/or `libgdal` on your system.
Has proven finicky to install.

Python version: >=3.10

Set up your Python environment as follows (prefered way; Dockerfile provided too):
```bash
pip install --upgrade pip
conda install -c conda-forge gdal
pip install rasterio opencv-python tqdm numpy scikit-learn wandb tmuxp tabulate
```
PyTorch __with__ GPU support (_preferred_):
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```
PyTorch __without__ GPU support:
```bash
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

Finally (keep this last), install lucidrains' vector quantization lib:
```bash
pip install vector-quantize-pytorch
```

## TODO's

- [] Provide the option to use Residual VQ (from the SoundStream paper,
which proposes to employ several quantizers to reccursively quantize
what's encoded, organized in a hierarchical fashion)
    - [] Test out the integrated Residual VQ (debug+benchmark)

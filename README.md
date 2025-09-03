# **Pan Cancer Image Analysis**

# Dependencies

- conda create -n PanCIA python=3.9.19
- conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia
- conda install -c conda-forge openslide
- pip install -r requirements.txt

If GCC is too old (need GCC 9 or later):
- conda install -c conda-forge gcc_linux-64 gxx_linux-64
- ln -sf $CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc $CONDA_PREFIX/bin/gcc
- ln -sf $CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++ $CONDA_PREFIX/bin/g++

# Remark
- Loads LoRA-based BiomedParse multiple times in a loop can get different results due to the nondeterminism in language encoder, so just loads it once.

# Acknowlegement
This is based on [BiomedParse](https://github.com/microsoft/BiomedParse) and [tiatoolbox](https://github.com/TissueImageAnalytics/tiatoolbox)
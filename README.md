# TLRM-SLDM Project

This repository contains the implementation of the TLRM project. 

## Dataset

The TLRM dataset can be accessed from [Hugging Face](https://huggingface.co/datasets/Rane7/TLRM_Dataset).

## Create a Conda Environment

To set up the environment required for this project, use the following commands:

```bash
conda env create -f environment.yaml
conda activate TlrmLsdm
```

## VAE Pretrained Weights

Download the VAE pretrained weights from the official [Hugging Face VAE weights](https://huggingface.co/sd-vae-ft-mse).

## Testing

Run the following command:

```bash
python test.py --datasetroot ./TLRM_dataset
```
## Acknowledgements

Our code is developed based on [SDM](https://github.com/WeilunWang/semantic-diffusion-model). Thanks [guided-diffusion](https://github.com/openai/guided-diffusion), "test_with_FID.py" in [OASIS](https://github.com/boschresearch/OASIS) for FID computation, and "lpips.py" in [stargan-v2](https://github.com/clovaai/stargan-v2) for LPIPS computation.



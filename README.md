# Attention_Unet
 
[![License](https://img.shields.io/github/license/EdgarLefevre/Attention_Unet?label=license)](https://github.com/EdgarLefevre/Attention_Unet/blob/main/LICENSE)

Implementation of attention Unet [Attention U-Net:
Learning Where to Look for the Pancreas].

## Installation

```shell
conda env create -f env.yml
```

## Usage

```shell
python -m Attention_Unet.train
```

## Results 

Attention map :
![Attention map](./data/att_map.png)

Prediction :
![Prediction](./data/prediction_wo_att.png)

## TODOs

 - [ ] bug unet ++ with attention (layer names)
 - [ ] postprocessing
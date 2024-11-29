# <img src="./images/lyra.svg" alt="icon" width="25" height="25"> Lyra: An Efficient and Speech-Centric Framework <br> for Omni-Cognition

<div align=center>
<img width="100%" src="images/teaser.png"/>
</div>
Overview of Lyra. It shows superiority compared with leading omni-models in:

1. Stronger performance.  
2. More versatile. 
3. More efficient.

## Release
- [11/29] 🔥 Lyra is coming! We release the [paper](https://arxiv.org/pdf/2403.18814.pdf), [demo](http://103.170.5.190:7860/), [code](https://github.com/dvlab-research/MGM), [models](https://huggingface.co/collections/YanweiLi/mgm-6603c50b9b43d044171d0854'). More training data and checkpoints will be released soon!

## Contents
- [Demo](#demo)
- [Install](#install)
- [Model](#model)
- [Preparation](#preparation)
- [Train](#train)
- [Evaluation](#evaluation)
- [Examples](#examples)
- [Citation](#citation)
- [Acknowledgement](#acknowledgement)
- [License](#license)

## Demo
We provide some selected examples in this section. More examples can be found in our [project page](https://mini-gemini.github.io/). Feel free to try our online [demo](http://103.170.5.190:7860/)!

<div align=center>
<img width="100%" src="images/teaser.png"/>
</div>

## Install
Please follow the instructions below to install the required packages.

NOTE: If you want to use the 2B version, please ensure to install the latest version Transformers (>=4.38.0).

1. Clone this repository
```bash
git clone https://github.com/dvlab-research/Lyra.git
```

2. Install Package
```bash
conda create -n lyra python=3.10 -y
conda activate lyra
cd Lyra
pip install --upgrade pip
pip install -e .
```

## Model

<div align=center>
<img width="98%" src="images/pipeline.png"/>
</div>
Lyra supports multi-modal inputs. When the data contains a speech modality, we use the latent cross-modality regularizer to assist. Data from each modality is processed through encoders and projectors before being sent into the LLM. Within the LLM, multi-modality LoRA and latent multi-modality extraction modules operate synergistically, facilitating the simultaneous generation of both speech and text outputs.


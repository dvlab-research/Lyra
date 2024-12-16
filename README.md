# <img src="assets/lyra.svg" alt="icon" width="30" height="30"> <span style="font-size:30px;">Lyra: An Efficient and Speech-Centric Framework <br>for Omni-Cognition</span>

<a href='https://huggingface.co/papers/2412.09501'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Discussion-orange'></a>
<a href='https://huggingface.co/collections/zszhong/lyra-model-674ea5bb3b39ff8f15de75fc'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-blue'></a>
<a href='https://huggingface.co/collections/zszhong/lyra-data-675d80fbab80334eb52cdd82'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Data-green'></a>
<a href='https://huggingface.co/collections/zszhong/lyra-evaluation-675d7f038747ba865932a149'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Evaluation-yellow'></a><br><a href='https://arxiv.org/pdf/2412.09501.pdf'><img src='https://img.shields.io/badge/Paper-arXiv-red'></a>
<a href='https://103.170.5.190:17860/'><img src='https://img.shields.io/badge/Project-Demo-violet'></a>
<a href='https://lyra-omni.github.io/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>


Overview of Lyra:

<div align=center>
<img width="98%" src="assets/overview.png"/>
</div>

Lyra shows superiority compared with leading omni-models in:
1. Stronger performance: Achieve SOTA results across a variety of speech-centric tasks.
2. More versatile:  Support image, video, speech/long-speech, sound understanding and speech generation.
3. More efficient: Less training data, support faster training and inference.

## Release
- [12/12] 🔥 Lyra is coming! We release the [paper](https://arxiv.org/pdf/2412.09501.pdf), [demo](https://103.170.5.190:17860/), [code](https://github.com/dvlab-research/Lyra), [models](https://huggingface.co/collections/zszhong/lyra-model-674ea5bb3b39ff8f15de75fc), [training data](https://huggingface.co/collections/zszhong/lyra-data-675d80fbab80334eb52cdd82) and [evaluation data](https://huggingface.co/collections/zszhong/lyra-evaluation-675d7f038747ba865932a149). More related checkpoints will be released soon!

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
We provide [video demo](https://www.youtube.com/watch?v=7kh-M0jmmtI) here for better experience and illustrations. More examples can be found in our [project page](https://lyra-omni.github.io/) and feel free to try our [online demo](https://103.170.5.190:17860/)! Due to the computing cost, GPU memory of the demo machine (GeForce RTX 3090), and uploading storage, the long-speech function is not supported for the current online demo. 😰

<p align="center" width="98%">
  <a href="https://youtu.be/7kh-M0jmmtI" target="_blank">
    <img src="https://raw.githubusercontent.com/dvlab-research/Lyra/main/assets/video.png" alt="Lyra" style="width: 98%; min-width: 300px; display: block; margin: auto;">
  </a>
</p>



## Install
Please follow the instructions below to install the required packages.

1. Clone this repository:
```bash
git clone https://github.com/dvlab-research/Lyra.git
```

2. Install Package:
```bash
conda create -n lyra python=3.10 -y
conda activate lyra
cd Lyra
pip install --upgrade pip
pip install -e .
```

3. Install optional packages for simultaneous text-speech generation:
```bash
pip install pip==24.0
pip install fairseq==0.12.2
pip install --upgrade pip
```

## Model

<div align=center>
<img width="98%" src="assets/framework.png"/>
</div>


Lyra supports multi-modal inputs. When the data contains a speech modality, we use the **latent cross-modality regularizer** to assist. Data from each modality is processed through encoders and projectors before being sent into the LLM. Within the LLM, **multi-modality LoRA** and l**atent multi-modality extraction** modules operate synergistically, facilitating the **simultaneous generation** of both speech and text outputs.

We provide all our fully finetuned models:

| Model        | Base LLM           | Vision Encoder     | Speech Encoder                                               | Projector   | Full                                                   |
| ------------ | ------------------ | ------------------ | ------------------------------------------------------------ | ----------- | ------------------------------------------------------ |
| Lyra_Mini_3B | [Qwen2VL_2B_LLM]() | [Qwen2VL_2B_ViT]() | [whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) | [3B_proj]() | [3B_ckpt]()                                            |
| Lyra_Base_9B | [Qwen2VL_7B_LLM]() | [Qwen2VL_7B_ViT]() | [whisper-large-v3](https://huggingface.co/openai/whisper-large-v3) | [9B_proj]() | [9B_ckpt](https://huggingface.co/zszhong/Lyra_Base_9B) |
| Lyra_Pro_74B | Qwen2VL_70B_LLM    | Qwen2VL_70B_ViT    | whisper-large-v3                                             | 74B_proj    | 74B_ckpt                                               |

## Preparation
### Training Data

We provide the processed data for the model training. All speech-related training data can be downloaded [Lyra-Data](https://huggingface.co/collections/zszhong/lyra-data-675d80fbab80334eb52cdd82).

For **model pretraining data**, please download the following the training multi-modality data and organize them as:

`⇒` means put the data in the local folder. The pretraining json file can be downloaded from [Lyra_Pretrain](https://huggingface.co/datasets/zszhong/Lyra-Data/tree/main/Lyra_Pretrain).

- [LibriSpeech](https://www.openslr.org/12) ⇒ `data/Lyra_Pretrain/LibriSpeech` 

  ​              and ⇒ `data/Lyra_SFT/multi_modality_speech/LibriSpeech`

  ​	      and ⇒ `data/Lyra_Eval/LibriSpeech`   download all training and develop data.

- [Common Voice](https://commonvoice.mozilla.org/en/datasets) ⇒ `data/Lyra_Pretrain/CommonVoice` download the English Common Voice Corpus.

During the pretraining process, we filtered out some noisy and short audio speech data.

For the **image part of finetuning data**, similar to Mini-Gemini, please download the following the instruction data and organize them as:

`⇒` means put the data in the local folder.

- [COCO train2017](http://images.cocodataset.org/zips/train2017.zip) ⇒ `data/Lyra_SFT/multi_modality_image/coco`
- [GQA](https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip) ⇒ `data/Lyra_SFT/multi_modality_image/gqa`
- [OCR-VQA](https://drive.google.com/drive/folders/1_GYPY5UkUy7HIcR0zq3ZCFgeZN7BAfm_?usp=sharing) (**we save all files as `.jpg`**) ⇒ `data/Lyra_SFT/multi_modality_image/ocr_vqa`
- [TextVQA](https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip) (not included for training) ⇒ `data/Lyra_SFT/multi_modality_image/textvqa`
- [VisualGenome part1](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip), [VisualGenome part2](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip) ⇒ `data/Lyra_SFT/multi_modality_image/vg`
- [ShareGPT4V-100K](https://github.com/InternLM/InternLM-XComposer/blob/main/projects/ShareGPT4V/docs/Data.md) ⇒ `data/Lyra_SFT/multi_modality_image/sam`, `share_textvqa`, `wikiart`, ...
- [LAION GPT4V](https://huggingface.co/datasets/laion/gpt4v-dataset) ⇒ `data/Lyra_SFT/multi_modality_image/gpt4v-dataset`
- [ALLaVA Instruction](https://github.com/FreedomIntelligence/ALLaVA) ⇒ `data/Lyra_SFT/multi_modality_image/ALLaVA-4V`
- [DocVQA](https://www.docvqa.org/datasets/docvqa) ⇒ `data/Lyra_SFT/multi_modality_image/docvqa`
- [ChartQA](https://github.com/vis-nlp/ChartQA) ⇒ `data/Lyra_SFT/multi_modality_image/chartqa`
- [DVQA](https://github.com/kushalkafle/DVQA_dataset) ⇒ `data/Lyra_SFT/multi_modality_image/dvqa`
- [AI2D](https://allenai.org/data/diagrams) ⇒ `data/Lyra_SFT/multi_modality_image/ai2d`

For the **audio part of finetuning data**, please download the following the instruction data and organize them as:

`⇒` means put the data in the local folder.

- [Lyra_MultiModal](https://huggingface.co/datasets/zszhong/Lyra-Data/tree/main/Lyra_SFT/multi_modality_speech) ⇒ `data/Lyra_SFT/multi_modality_speech/Lyra_MM` 

  For reproduced details, please refer the [Lyra multi-modality preparation](https://github.com/dvlab-research/Lyra/tree/main/data_preparation/multi_modality).

For the **long speech** audio finetuning data, please download the following the instruction data and organize them as:

`⇒` means put the data in the local folder.

- [Lyra_LongSpeech](https://huggingface.co/datasets/zszhong/Lyra-Data/tree/main/Lyra_SFT/long_speech) ⇒ `data/Lyra_SFT/long_speech/Lyra_LongSpeech`

  For reproduced details, please refer the [Lyra long-speech preparation](https://github.com/dvlab-research/Lyra/tree/main/data_preparation/long_speech).

For the **text-speech generation** data, please download the following the instruction data and organize them as:

`⇒` means put the data in the local folder.

- [Lyra_SpeechGeneration](https://huggingface.co/datasets/zszhong/Lyra-Data/tree/main/Lyra_SFT/speech_generation)  ⇒ `data/Lyra_SFT/speech_generation` 

  For reproduced details, please refer the [Lyra speech generation preparation](https://github.com/dvlab-research/Lyra/tree/main/data_preparation/speech_generation).

### Evaluation Data

All speech-related evaluation data can be downloaded [Lyra-Evaluation](https://huggingface.co/collections/zszhong/lyra-evaluation-675d7f038747ba865932a149).

For **speech-centric evaluation data**, we mainly consider three types:

1. **text-speech ability**: LibriSpeech, Lyra_needle_in_a_haystack

- [Lyra_needle_in_a_haystack](https://huggingface.co/datasets/zszhong/Lyra-Eval/tree/main/Lyra_needle_in_a_haystack) ⇒ `data/Lyra_Eval/Lyra_needle_in_a_haystack` 

2. **image-speech ability**: TextVQA_speech, MM_vet_speech, Docvqa_val, Chartvqa_human

- [TextVQA_speech](https://huggingface.co/datasets/zszhong/Lyra-Eval/tree/main/TextVQA_speech) ⇒ `data/Lyra_Eval/TextVQA_speech`

- [MM_vet_speech](https://huggingface.co/datasets/zszhong/Lyra-Eval/tree/main/MM_vet_speech) ⇒ `data/Lyra_Eval/MM_vet_speech`

- [Docvqa_val](https://huggingface.co/datasets/zszhong/Lyra-Eval/tree/main/Docvqa_val) ⇒ `data/Lyra_Eval/Docvqa_val`

- [Chartvqa_human](https://huggingface.co/datasets/zszhong/Lyra-Eval/tree/main/Chartvqa_human) ⇒ `data/Lyra_Eval/Chartvqa_human`

3. **video-speech ability**: VideoMME_speech

- [VideoMME_speech](https://huggingface.co/datasets/zszhong/Lyra-Eval/tree/main/VideoMME_speech) ⇒ `data/Lyra_Eval/VideoMME_speech`


Please put the pretrained data, finetuned data, and eval data in  `Lyra_Pretrain`, `Lyra_SFT`, and `Lyra_Eval` subset following [Structure](#structure).

### Pretrained Weights

We recommend users to download the pretrained weights from the following link:

Qwen2VL_XB_LLM and Qwen2VL_XB_ViT are extracted from [Qwen2-VL](https://github.com/QwenLM/Qwen2-VL) to adapt to our training framework. 

For your convenience we also provide the corresponding download links in the [Model](#model) part.

[whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo), [whisper-large-v3](https://huggingface.co/openai/whisper-large-v3),  [imagebind_huge](https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth), and put them in `model_zoo` following [Structure](#structure).

Download the unit-based HiFi-GAN vocoder using the follow commands:

```shell
wget https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/mhubert_vp_en_es_fr_it3_400k_layer11_km1000_lj/g_00500000 -P model_zoo/audio/vocoder/
wget https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/mhubert_vp_en_es_fr_it3_400k_layer11_km1000_lj/config.json -P model_zoo/audio/vocoder/
```

### Structure

The folder structure should be organized as follows before training.

```
Lyra
├── lyra
├── scripts
├── work_dirs
│   ├── Lyra
│   │   ├── Lyra_Mini_3B
│   │   ├── Lyra_Base_9B
│   │   ├── Lyra_Pro_74B
│   │   ├── ...
├── model_zoo
│   ├── LLM
│   │   ├── Qwen2VL_2B_LLM
│   │   ├── Qwen2VL_7B_LLM
│   │   ├── Qwen2VL_70B_LLM
│   │   ├── Qwen2.5
│   │   ├── LLaMA3.2
│   │   ├── ...
│   ├── vision
│   │   ├── Qwen2VL_2B_ViT
│   │   ├── Qwen2VL_7B_ViT
│   │   ├── Qwen2VL_70B_ViT
│   │   ├── clip-vit-large
│   │   ├── siglip
│   │   ├── ConvNeXt
│   │   ├── ...
│   ├── audio
│   │   ├── whisper-large-v3-turbo
│   │   ├── whisper-large-v3
│   │   ├── imagebind_huge
│   │   ├── vocoder
│   │   ├── ...
├── data
│   ├── Lyra_Pretrain
│   │   ├── lyra_pretrain.json
│   │   ├── LibriSpeech
│   │   ├── CommonVoice
│   ├── Lyra_SFT
│   │   ├── multi_modality_speech
│   │   │   ├── lyra_multimodal.json
│   │   │   ├── Lyra_MM
│   │   │   ├── LibriSpeech
│   │   ├── multi_modality_image (similar to MGM-Finetune)
│   │   │   ├── llava
│   │   │   ├── coco
│   │   │   ├── gqa
│   │   │   ├── ocr_vqa
│   │   │   ├── textvqa
│   │   │   ├── vg
│   │   │   ├── gpt4v-dataset
│   │   │   ├── ...
│   │   ├── long_speech
│   │   │   ├── lyra_longspeech.json
│   │   │   ├── Lyra_LongSpeech
│   │   ├── speech_generation
│   │   │   ├── lyra_speechgeneration.json
│   ├── Lyra_Eval
│   │   ├── LibriSpeech
│   │   ├── TextVQA_speech
│   │   ├── MM_vet_speech
│   │   ├── Docvqa_val
│   │   ├── Chartvqa_human
│   │   ├── VideoMME_speech
│   │   ├── Lyra_needle_in_a_haystack
```

## Train

The training process consists of four stages: (1) feature alignment stage: bridge the speech and language tokens; (2) multi-modality instruction tuning stage: teach the model to follow text-image-speech multimodal instructions. (3) long-speech instruction tuning stage: enable the model to handle long speech audios. (4) text-speech streaming generation stage: Enable the model to stream both text and speech simultaneously.

Our models are trained on 8 A100 GPUs with 80GB memory. To train on fewer GPUs, you can reduce the `per_device_train_batch_size` and increase the `gradient_accumulation_steps` accordingly. Always keep the global batch size the same: `per_device_train_batch_size` x `gradient_accumulation_steps` x `num_gpus`.

Please make sure you download and organize the data following [Preparation](#preparation) before training.

NOTE: Please set `hostfile/hostfile_2` for 2 machine training and `hostfile/hostfile_4` for 4 machine training.

 (1) feature alignment stage: 

```bash
bash scripts/train/Lyra_Base_9B/Lyra_Base_qwen2vl_9B_Pretrain.sh
```
 (2) multi-modality instruction tuning stage:
```bash
bash scripts/train/Lyra_Base_9B/Lyra_Base_qwen2vl_9B_SFT_text_image_speech.sh
```
(3) long-speech instruction tuning stage:
```bash
bash scripts/train/Lyra_Base_9B/Lyra_Base_qwen2vl_9B_SFT_long_speech.sh
```
(4) text-speech streaming generation stage:

```bash
bash scripts/train/Lyra_Base_9B/Lyra_Base_qwen2vl_9B_SFT_speech_generate.sh
```

## Evaluation

### Benchmarks Results

<table>
  <tr>
    <th rowspan="2">Omni Comparison</th>
    <th rowspan="2">Params.</th>
    <th colspan="3" align="center">Text-Image</th>
    <th colspan="3" align="center">Text-Video</th>
    <th colspan="3" align="center">Image-Speech</th>
    <th rowspan="1">Text-Speech</th>
  </tr>
  <tr>
    <th>TextVQA</th>
    <th>MME</th>
    <th>MM-Vet</th>
    <th>VideoMME</th>
    <th>MVBench</th>
    <th>Egoschema</th>
    <th>TextVQA<sup>s</sup></th>
    <th>DocVQA<sup>s</sup></th>
    <th>ChartQA<sup>s</sup></th>
    <th>LibriSpeech</th>
  </tr>
  <tr>
    <td>Mini-Gemini</td>
    <td>8B</td>
    <td>71.9</td>
    <td>1989</td>
    <td>53.5</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>LLaVA-OV</td>
    <td>7B</td>
    <td>65.4</td>
    <td>1998</td>
    <td>57.5</td>
    <td>58.2</td>
    <td>56.7</td>
    <td>60.1</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>Intern-VL2</td>
    <td>8B</td>
    <td>77.4</td>
    <td>2211</td>
    <td>60.0</td>
    <td>54.0</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>Mini-Omni</td>
    <td>7B</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>4.5</td>
  </tr>
  <tr>
    <td>SALMONN</td>
    <td>13B</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>2.1</td>
  </tr>
  <tr>
    <td>Qwen2-Audio</td>
    <td>8B</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>1.6</td>
  </tr>
  <tr>
    <td>Intern-Omni</td>
    <td>8B</td>
    <td>80.6</td>
    <td>2210</td>
    <td>60.0</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>69.1</td>
    <td>79.9</td>
    <td>56.0</td>
    <td>-</td>
  </tr>
  <tr>
    <td>VITA</td>
    <td>66B</td>
    <td>-</td>
    <td>2097</td>
    <td>41.6</td>
    <td>59.2</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>8.1</td>
  </tr>
  <tr>
    <td>EMOVA</td>
    <td>14B</td>
    <td>82.0</td>
    <td>2205</td>
    <td>55.8</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>4.0</td>
  </tr>
  <tr>
    <td><b>Lyra-Mini</b></td>
    <td>3B</td>
    <td>78.3</td>
    <td>1884</td>
    <td>51.2</td>
    <td>55.0</td>
    <td>62.5</td>
    <td>54.1</td>
    <td>73.9</td>
    <td>75.0</td>
    <td>40.7</td>
    <td>2.1</td>
  </tr>
  <tr>
    <td><b>Lyra-Base</b></td>
    <td>9B</td>
    <td>82.6</td>
    <td>2335</td>
    <td>63.5</td>
    <td>62.8</td>
    <td>67.2</td>
    <td>63.2</td>
    <td>80.0</td>
    <td>85.5</td>
    <td>61.0</td>
    <td>2.0</td>
  </tr>
  <tr>
    <td><b>Lyra-Pro</b></td>
    <td>74B</td>
    <td>83.5</td>
    <td>2485</td>
    <td>71.4</td>
    <td>69.9</td>
    <td>72.3</td>
    <td>75.8</td>
    <td>81.0</td>
    <td>89.4</td>
    <td>68.5</td>
    <td>1.8</td>
  </tr>
</table>


### Benchmarks scripts



Please make sure you download and organize the [evaluation data](https://huggingface.co/collections/zszhong/lyra-evaluation-675d7f038747ba865932a149) following [Preparation](#preparation) before starting evaluation.

We provide four speech **speech-centric** evaluation benchmark scripts here:

**Text-speech ability**: LibriSpeech:

```bash
# you can change the model path and lora path in the script:
# CKPT="Lyra_Base_9B", LORA_PATH="Lyra_Base_9B/speech_lora"
# the LibriSpeech test-clean WER result of Lyra-Base-9B is about 2.0%
bash scripts/eval/lyra_librispeech_wer.sh
```

**Image-speech ability**: TextVQA_speech:

```bash
# the TextVQA (speech) accuracy result of Lyra-Base-9B is about 80.5%
bash scripts/eval/lyra_textvqa_speech.sh
```

**Image-speech ability**: Chartvqa_human:

```bash
# the ChartQA (speech) accuracy result of Lyra-Base-9B is about 61.0%
bash scripts/eval/lyra_chartvqa_speech.sh
```

**Image-speech ability**: Docvqa_val:

```bash
# the DocVQA (speech) accuracy result of Lyra-Base-9B is about 86.2%
bash scripts/eval/lyra_docvqa_speech.sh
```



### CLI Inference

Chat with images without the need of Gradio interface. It also supports multiple GPUs, 4-bit and 8-bit quantized inference.
Please make sure you have installed [fairseq](https://github.com/facebookresearch/fairseq) for speech generation, and try the following command for speech and generation inference:

```bash
# image-file:       <path to your image: context>
# speech-file:      <path to your audio: instruction>
# generate speech:  <output path to generated speech: examples/pred_roundX.wav>
python -m lyra.serve.cli \
	--model-path work_dirs/Lyra_Base_9B \
	--image-file examples/Chinese_painting.jpg \
	--audio-file examples/Chinese_painting.mp3 \
	--generate-speech
```

Lyra can also handle your long speech input (max duration can be about two or three hours, suggest on A100 GPUs).

Here is an example: [ABC New, Oct. 1, 2024](https://www.youtube.com/watch?v=A7LTOsf7JMQ&t=1063s), 20 mins:

```bash
# speech-file: <path to your long audio: context>
# instuction by the text keyboard input
python -m lyra.serve.cli \
	--model-path work_dirs/Lyra_Base_9B \
	--audio-file examples/ABC_News_20241001.mp3 \
	--generate-speech
```

Here is an example for video input with its audio (you can use [ffmpeg](https://github.com/kkroening/ffmpeg-python) or other tools to extract video's audio): 

```bash
# video-file:  <path to your video: context>
# speech-file: <path to your audio: instruction>
python -m lyra.serve.cli \
	--model-path work_dirs/Lyra_Base_9B \
	--video-file examples/movement.mp4 \
	--audio-file examples/movement.mp3 \
	--generate-speech
```

Here is an example for video input and text instruction:

```bash
# video-file: <path to your video: context>
# instuction by the text keyboard input
python -m lyra.serve.cli \
	--model-path work_dirs/Lyra_Base_9B \
	--video-file examples/Trump.mp4 \
	--generate-speech
```



### Gradio Web UI

To be release soon!



## Examples
We provide some examples in this section. More examples can be found in our [project page](https://lyra-omni.github.io/).

<div align=center>
<img width="98%" src="assets/demo_vlm.png"/>
</div>

<div align=center>
<img width="98%" src="assets/demo_news.png"/>
</div>


## Citation
If you find this repo useful for your research, please consider citing the paper😊:
```
@article{zhong2024lyra,
  title={Lyra: An Efficient and Speech-Centric Framework for Omni-Cognition},
  author={Zhong, Zhingsheng and Wang, Chengyao and Liu, Yuqi and Yang, Senqiao and Tang, Longxiang and Zhang, Yuechen and Li, Jingyao and Qu, Tianyuan and Li, Yanwei and Chen, Yukang and Yu, Shaozuo and Wu, Sitong and Lo, Eric and Liu, Shu and Jia, Jiaya},
  journal={arXiv preprint arXiv:2412.09501},
  year={2024}
}
```

## Acknowledgement
We would like to thank the following repos for their great work:

- This work is built upon the [LLaVA Series](https://github.com/LLaVA-VL/LLaVA-NeXT), [Mini-Gemini](https://github.com/dvlab-research/MGM), [LLaMA-Omni](https://github.com/ictnlp/LLaMA-Omni), [fairseq](https://github.com/facebookresearch/fairseq), [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval).
- This work utilizes models from [Qwen2-VL](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct), [Qwen2 Series](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct), [LLaMA3 Series](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision), and [Whisper](https://huggingface.co/openai/whisper-large-v3).

## License
[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-yellow.svg)](https://github.com/dvlab-research/Lyra/blob/main/LICENSE)
[![Data License](https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-orange.svg)](https://github.com/dvlab-research/Lyra/blob/main/DATA_LICENSE)
[![Weight License](https://img.shields.io/badge/Weight%20License-CC%20By%20NC%204.0-red)](https://github.com/dvlab-research/Lyra/blob/main/WEIGHT_LICENSE)

The data and checkpoint is intended and licensed for research use only. They are also restricted to uses that follow the license agreement of LLaVA, Qwen, LLaMA, Whisper, and GPT-4o. The dataset is CC BY NC 4.0 (allowing only non-commercial use) and models trained using the dataset should not be used outside of research purposes.
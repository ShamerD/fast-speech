# fast-speech

Implementation of
[FastSpeech](https://arxiv.org/pdf/1905.09263.pdf)
Text-to-Speech model trained on LJSpeech.
 
## Installation
```shell
git clone https://github.com/NVIDIA/waveglow.git
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html

# load model weights
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1-YofXqcfDIk2SZZYRA0KJbi7NRs5xRZC' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1-YofXqcfDIk2SZZYRA0KJbi7NRs5xRZC" -O resources/fastspeech.pth && rm -rf /tmp/cookies.txt
```

## Usage
#### Train:
```shell
python3 train.py -c <config_file> [-r <resume_checkpoint>] [--lr <learning_rate>] [--bs <batch_size>]
```

#### Inference:

Input for inference is a text file with source sentences located in separate lines.
If not provided default samples will be used.

```shell
python3 inference.py -c <config_file> -r <checkpoint> [-s <source_file>] [-t <target_directory>]
```

For example :
```shell
python3 inference.py -c configs/main.json -r resources/fastspeech.pth
```

Default samples:
* `A defibrillator is a device that gives a high energy electric shock to the heart of someone who is in cardiac arrest`
* `Massachusetts Institute of Technology may be best known for its math, science and engineering education`
* `Wasserstein distance or Kantorovich Rubinstein metric is a distance function defined between probability distributions on a given metric space`

## Project structure
* `configs/` contains configs which were used to train model
* `data/` contains data (LJSpeech downloads there by default) and trainval split
* `notebooks` contains notebooks which show how the model was trained
* `src` contains source codes
* `train.py` is a training script (it downloads all needed data if it is not present)
* `inference.py` is an inference script which takes text file and outputs audio files in a directory
# fast-speech

Implementation of
[FastSpeech](https://arxiv.org/pdf/1905.09263.pdf)
Text-to-Speech model trained on LJSpeech.
 
## Installation
```shell
git clone https://github.com/NVIDIA/waveglow.git
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html

# load model weights
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1-YofXqcfDIk2SZZYRA0KJbi7NRs5xRZC' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1-YofXqcfDIk2SZZYRA0KJbi7NRs5xRZC" -O default_test/checkpoint.pth && rm -rf /tmp/cookies.txt
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
python3 inference.py -c <config_file> -r <checkpoint> -s <source_file> -t <target_directory>
```

Default samples:
* `A defibrillator is a device that gives a high energy electric shock to the heart of someone who is in cardiac arrest`
* `Massachusetts Institute of Technology may be best known for its math, science and engineering education`
* `Wasserstein distance or Kantorovich Rubinstein metric is a distance function defined between probability distributions on a given metric space`

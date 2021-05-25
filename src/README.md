### Description of source code
- **util.py**: functions to load and split dataset
- **seq2seq.py**: class for simple RNN based sequence-to-sequence model
- **attention.py**: class for sequence-to-sequence with attention
- **transformer.py**: class for transformer
- **models.py**: functions to train and test models
- **ddrfuzz.py**: main file

### Prerequisite
- python: 3.6.9
- numpy: 1.18.2
- scikit-learn: 0.24.2
- tensorflow: 2.3.0
```
pip install numpy==1.18.2
pip install tensorflow==2.3.0
pip install scikit-learn==0.24.2
```
### Usage
```
python3 ./ddrfuzz.py --model [seq2seq|attention|transformer] --maxlen [int] --emb_dim [int] --batch_size [int] --epochs [int] --patience [int] --path [data directory]
```
- model: [seq2seq|attention|transformer] (default: attention)
- maxlen: max length of input data (default: 500)
- emb_dim: dimension of embedding layer (default: 64)
- batch_size: batch size of training (default: 16)
- epochs: training loop (default: 20)
- patience: patience of earlystopping (default: None)
- path: source directory (default: '../seq2seq/data/')
# image-gpt

## Hot to use

### Installation

```bash
pip install -r requirements.txt
```

### Training

```bash
python main.py --dataset cifar10 train configs/train.yml
```

### Test FID score

Download pretrained model from this link
https://drive.google.com/file/d/1zfsSf9e5RuaQH6NAeq--zeJsKSi0o7nu/view?usp=sharing

Put the model under `model` directory

```bash
python main.py --dataset cifar10 test model/pretrained.ckpt configs/test.yml
```

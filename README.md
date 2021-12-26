# image-gpt

## Hot to use

### Training

```bash
python main.py --dataset cifar10 train configs/train.yml
```

### Test FID score

Download pretrained model from this link
https://drive.google.com/file/d/1EUoTzfzA2rXojbviAUagLeT4AAukhknN/view?usp=sharing

Put the model under `model` directory

```bash
python main.py --dataset cifar10 test model/pretrained.ckpt configs/test.yml
```

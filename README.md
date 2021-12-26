# image-gpt

## Hot to use

### Training

```bash
python main.py --dataset cifar10 train configs/train.yml
```

#### Test FID score

Download pretrained model from this link
Put the model under `model` directory

```bash
python main.py --dataset cifar10 test model/pretrained.ckpt configs/test.yml
```

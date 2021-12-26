# image-gpt

## Hot to use

### Training

```bash
python run.py --dataset cifar10 train configs/train.yml
```

#### Test FID score

```bash
python run.py --dataset cifar10 test model/pretrained.ckpt configs/test.yml
```

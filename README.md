# image-gpt

## Hot to use

### Training

```bash
python main.py --dataset cifar10 train configs/train.yml
```

#### Test FID score

```bash
python main.py --dataset cifar10 test model/pretrained.ckpt configs/test.yml
```

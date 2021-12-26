# image-gpt

## Hot to use

### Training

Models can be trained using `src/run.py` with the `train` subcommand. 

```bash
python run.py --dataset cifar10 train configs/train.yml
```

#### Test FID score

```bash
python run.py --dataset cifar10 test model/pretrained.ckpt configs/test.yml
```

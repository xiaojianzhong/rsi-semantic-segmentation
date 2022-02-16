# Remote Sensing Imagery Semantic Segmentation

Read this in other languages: English | [简体中文](README_zh-CN.md)

## Table of Contents

- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Configurations](#configurations)
  - [Configuration Name Format](#configuration-name-format)
- [Supported Datasets](#supported-datasets)
- [Supported Models](#supported-models)
- [Usage](#usage)
    - [Train](#train)
    - [Test](#test)
    - [Inference](#inference)
- [License](#license)

## <a name="project-structure"></a> Project Structure

```
rsi-semantic-segmentation
  |---- configs
  |       |---- __init__.py
  |       |---- gf2-building_deeplabv3-resnet50_ce_adam_plateau_8_0.001_40.yaml
  |       |---- massachusetts-building_deeplabv3-resnet50_dice_adam_plateau_8_0.001_40.yaml
  |
  |---- criterions
  |       |---- __init__.py
  |       |---- bce.py
  |       |---- ce.py
  |       |---- dice.py
  |
  |---- datas
  |       |---- __init__.py
  |       |---- base.py
  |       |---- gf2_building.py
  |       |---- massachusetts_building.py
  |       |---- transform.py
  |
  |---- models
  |       |---- decoders
  |       |       |---- __init__.py
  |       |       |---- deeplabv3.py
  |       |
  |       |---- encoders
  |       |       |---- __init__.py
  |       |       |---- resnet.py
  |       |
  |       |---- modules
  |       |       |---- __init__.py
  |       |       |---- aspp.py
  |       |
  |       |---- utils
  |       |       |---- init.py
  |       |
  |       |---- __init__.py
  |       |---- deeplabv3.py
  |
  |---- optimizers
  |       |---- __init__.py
  |
  |---- schedulers
  |       |---- __init__.py
  |
  |---- tools
  |       |---- datasets
  |               |---- massachusetts_building
  |                       |---- patch.py
  |
  |---- .gitignore
  |---- inference.py
  |---- LICENSE
  |---- metric.py
  |---- README.md
  |---- README_zh-CN.md
  |---- requirements.txt
  |---- test.py
  |---- train.py
```

## <a name="prerequisites"></a> Prerequisites

- [NumPy](https://numpy.org/) for multi-dimensional data representation on CPU
- [Pandas](https://pandas.pydata.org/) for parsing `.csv` files
- [scikit-image](https://scikit-image.org/) for reading, writing and showing images
- [tensorboardX](https://github.com/lanpa/tensorboardX) for logging to TensorBoard
- [timm](https://github.com/rwightman/pytorch-image-models) for computer vision backbones in PyTorch
- [PyTorch](https://pytorch.org/) for neural network representation and calculation
- [TorchVision](https://pytorch.org/vision/) for basic components applied in computer vision
- [tqdm](https://github.com/tqdm/tqdm) for drawing progress bar
- [yacs](https://github.com/rbgirshick/yacs) for parsing `.yaml` configuration files

All these Python third-party packages can be easily installed through `pip`:

```shell
$ pip install numpy pandas scikit-image tensorboardX timm torch torchvision tqdm yacs
```

## <a name="configurations"></a> Configurations

| dataset                  | method               | criterion | optimizer | scheduler | batch size | LR    | epochs | config                                                                                        |
|:------------------------:|:--------------------:|:---------:|:---------:|:---------:|:----------:|:-----:|:------:|:--------------------------------------------------------------------------------------------:|
| `gf2-building`           | `deeplabv3-resnet50` | `ce`      | `adam`    | `plateau` | 8          | 0.001 | 10     | [config](configs/gf2-building_deeplabv3-resnet50_ce_adam_plateau_8_0.001_10.yaml)             |
| `massachusetts-building` | `deeplabv3-resnet50` | `dice`    | `adam`    | `plateau` | 8          | 0.001 | 40     | [config](configs/massachusetts-building_deeplabv3-resnet50_dice_adam_plateau_8_0.001_40.yaml) |

### <a name="configuration-name-format"></a> Configuration Name Format

```
{dataset}_{method}_{criterion}_{optimizer}_{scheduler}_{batch size}_{lr}_{epochs}.yaml
```

- `{dataset}`: dataset name like `massachusetts-building`, `massachusetts-road`, etc.
- `{method}`: method name like `deeplabv3+resnet50`, `deeplabv3+resnet101`, etc.
- `{criterion}`: criterion name like `ce`, `bce`, etc.
- `{optimizer}`: optimizer name like `sgd`, `adam`, etc.
- `{scheduler}`: scheduler name like `poly`, `plateau`, etc.
- `{batch size}`: batch size during training, e.g. `4`, `8`.
- `{lr}`: basic learning rate for training, e.g. `0.01`, `0.001`.
- `{epochs}`: epochs for training, e.g. `20`, `40`.

## <a name="supported-datasets"></a> Supported Datasets

- [x] [GF2 Building](datas/gf2_building.py)
- [x] [Massachusetts Building](datas/massachusetts_building.py)

## <a name="supported-models"></a> Supported Models

- [x] [DeepLabV3 (ArXiv'2017)](models/deeplabv3.py)

## <a name="usage"></a> Usage

### <a name="train"></a> Train

```shell
$ python train.py configs/massachusetts-building_deeplabv3-resnet50_dice_adam_plateau_8_0.001_40.yaml \
                  --checkpoint ./best.pth \
                  --path ./runs/20211206-201700/ \
                  --no-validate \
                  --nodes 1 \
                  --gpus 1 \
                  --rank-node 0 \
                  --backend nccl \
                  --master-ip localhost \
                  --master-port 8888 \
                  --seed 42 \
                  --opt-level O0
```

- `config`: Configuration to be used, which must be specified.
- `--checkpoint`: Checkpoint to be loaded. Default: train from scratch.
- `--path`: Directory to save experiment output files. Default: a directory named by current time.
- `--no-validate`: Whether not to validate on the validation set during training. Default: do validation.
- `-n` / `--nodes`: Number of nodes / machines, should be `1` when training on a single machine. Default: `1`.
- `-g` / `--gpus`: Number of GPUs per node / machine. Default: `1`.
- `-r` / `--rank-node`: Ranking of the current node / machine, should be in range of `0` ~ `nodes-1`. Default: `0`.
- `--backend`: Backend for PyTorch DDP. Default: `nccl`.
- `--master-ip`: Network IP of the master node / machine. Default: `localhost`.
- `--master-port`: Network port of the master port on the master node / machine. Default: `8888`.
- `--seed`: Random seed. Default: `42`.
- `--opt-level`: Optimizing level for `nvidia/apex`. Default: `O0`.

### <a name="test"></a> Test

```shell
$ python test.py configs/massachusetts-building_deeplabv3-resnet50_dice_adam_plateau_8_0.001_40.yaml \
                 ./best.pth \
                 --device cuda:0
```

- `config`: configuration to be used, which must be specified.
- `checkpoint`: Checkpoint to be loaded, which must be specified.
- `--device`: Device for testing, could be either CPU or GPU. Default: GPU #0.

### <a name="inference"></a> Inference

```shell
$ python inference.py configs/massachusetts-building_deeplabv3-resnet50_dice_adam_plateau_8_0.001_40.yaml \
                      ./best.pth \
                      ./22828930_15.tif \
                      --output ./output.tif \
                      --device cuda:0 \
                      --no-show \
                      --no-save
```

- `config`: configuration to be used, which must be specified.
- `checkpoint`: Checkpoint to be loaded, which must be specified.
- `input`: Input image, which must be specified.
- `--output`: Output segmentation map. Default: output.tif
- `--device`: Device for inferring, could be either CPU or GPU. Default: GPU #0.
- `--no-show`: Whether not to show segmentation results. Default: do showing.
- `--no-save`: Whether not to save segmentation results. Default: do saving.

## <a name="license"></a> License

This project is released under the [MIT license](LICENSE).

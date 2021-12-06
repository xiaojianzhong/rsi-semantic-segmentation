# Remote Sensing Imagery Semantic Segmentation

Read this in other languages: English | [简体中文](README_zh-CN.md)

## Table of Contents

- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Configurations](#configurations)
- [Supported Datasets](#supported-datasets)
- [Supported Models](#supported-models)
- [License](#license)

## <a name="project-structure"></a> Project Structure

```
rsi-semantic-segmentation
  |---- configs
  |       |---- __init__.py
  |       |---- massachusetts-building_deeplabv3+resnet50_sigmoid+dice_adam_plateau_8_0.001_40.yaml
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
  |       |---- massachusetts_building.py
  |       |---- patch.py
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

| dataset                  | method               | criterion      | optimizer | scheduler | batch size | LR    | epochs | config                                                                                                |
|:------------------------:|:--------------------:|:--------------:|:---------:|:---------:|:----------:|:-----:|:------:|:-----------------------------------------------------------------------------------------------------:|
| `massachusetts-building` | `deeplabv3+resnet50` | `sigmoid+dice` | `adam`    | `plateau` | 8          | 0.001 | 40     | [config](configs/massachusetts-building_deeplabv3+resnet50_sigmoid+dice_adam_plateau_8_0.001_40.yaml) |

## <a name="supported-datasets"></a> Supported Datasets

- [x] [Massachusetts Building](datas/massachusetts_building.py)

## <a name="supported-models"></a> Supported Models

- [x] [DeepLabV3 (ArXiv'2017)](models/deeplabv3.py)

## <a name="license"></a> License

This project is released under the [MIT license](LICENSE).

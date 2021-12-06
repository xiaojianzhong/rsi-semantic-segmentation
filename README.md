# Remote Sensing Imagery Semantic Segmentation

Read this in other languages: English | [简体中文](README_zh-CN.md)

## Table of Contents

- [Prerequisites](#prerequisites)
- [Supported Datasets](#supported-datasets)
- [Supported Models](#supported-models)
- [License](#license)

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

## <a name="supported-datasets"></a> Supported Datasets

- [x] [Massachusetts Building](datas/massachusetts_building.py)

## <a name="supported-models"></a> Supported Models

- [x] [DeepLabV3 (ArXiv'2017)](models/deeplabv3.py)

## <a name="license"></a> License

This project is released under the [MIT license](LICENSE).

# 遥感图像语义分割

阅读其他语言的版本：[English](README.md) | 简体中文

## 目录

- [项目结构](#project-structure)
- [环境要求](#prerequisites)
- [配置](#configurations)
- [支持的数据集](#supported-datasets)
- [支持的模型](#supported-models)
- [用法](#usage)
    - [训练](#train)
    - [测试](#test)
    - [推理](#inference)
- [许可证](#license)

## <a name="project-structure"></a> 项目结构

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

## <a name="prerequisites"></a> 环境要求

- [NumPy](https://numpy.org/) 用于 CPU 上的多维数据表示
- [Pandas](https://pandas.pydata.org/) 用于解析 `.csv` 文件
- [scikit-image](https://scikit-image.org/) 用于读、写、显示图像
- [tensorboardX](https://github.com/lanpa/tensorboardX) 用于输出 TensorBoard 日志
- [timm](https://github.com/rwightman/pytorch-image-models) 用于提供 PyTorch 上的计算机视觉骨干网络支持
- [PyTorch](https://pytorch.org/) 用于神经网络表示与计算
- [TorchVision](https://pytorch.org/vision/) 用于提供计算机视觉基础组件支持
- [tqdm](https://github.com/tqdm/tqdm) 用于绘制进度条
- [yacs](https://github.com/rbgirshick/yacs) 用于解析 `.yaml` 配置文件

所有这些 PyThon 第三方包都可以简单地使用 `pip` 进行安装：

```shell
$ pip install numpy pandas scikit-image tensorboardX timm torch torchvision tqdm yacs
```

## <a name="configurations"></a> 配置

| dataset                  | method               | criterion      | optimizer | scheduler | batch size | LR    | epochs | config                                                                                                |
|:------------------------:|:--------------------:|:--------------:|:---------:|:---------:|:----------:|:-----:|:------:|:-----------------------------------------------------------------------------------------------------:|
| `massachusetts-building` | `deeplabv3+resnet50` | `sigmoid+dice` | `adam`    | `plateau` | 8          | 0.001 | 40     | [config](configs/massachusetts-building_deeplabv3+resnet50_sigmoid+dice_adam_plateau_8_0.001_40.yaml) |

配置文件名格式：`{dataset}_{method}_{criterion}_{optimizer}_{scheduler}_{batch size}_{lr}_{epochs}.yaml`

## <a name="supported-datasets"></a> 支持的数据集

- [x] [Massachusetts Building](datas/massachusetts_building.py)

## <a name="supported-models"></a> 支持的模型

- [x] [DeepLabV3 (ArXiv'2017)](models/deeplabv3.py)

## <a name="usage"></a> 用法

### <a name="train"></a> 训练

```shell
$ python train.py configs/massachusetts-building_deeplabv3+resnet50_sigmoid+dice_adam_plateau_8_0.001_40.yaml \
                  --checkpoint ./best.pth \
                  --device cuda:0 \
                  --path ./runs/20211206-201700/ \
                  --no-validate
```

- `config` 指定所使用的配置文件，不可省略
- `--checkpoint` 指定要加载的保存点，默认从零开始进行训练
- `--device` 指定训练时要使用的设备，可以是 CPU 或 GPU，默认为 0 卡
- `--path` 指定实验日志文件要存放到的路径，默认为一个以当前时间为名称的路径
- `--no-validate` 指定在训练过程中是否不在验证集上进行验证，默认进行验证

### <a name="test"></a> 测试

```shell
$ python test.py configs/massachusetts-building_deeplabv3+resnet50_sigmoid+dice_adam_plateau_8_0.001_40.yaml \
                 ./best.pth \
                 --device cuda:0
```

- `config` 指定所使用的配置文件，不可省略
- `checkpoint` 指定要加载的保存点，不可省略
- `--device` 指定测试时要使用的设备，可以是 CPU 或 GPU，默认为 0 卡

### <a name="inference"></a> 推理

```shell
$ python inference.py configs/massachusetts-building_deeplabv3+resnet50_sigmoid+dice_adam_plateau_8_0.001_40.yaml \
                      ./best.pth \
                      ./22828930_15.tif \
                      --output ./output.tif \
                      --device cuda:0 \
                      --no-show \
                      --no-save
```

- `config` 指定所使用的配置文件，不可省略
- `checkpoint` 指定要加载的保存点，不可省略
- `input` 指定输入图像，不可省略
- `--output`: 指定输出分割图的文件名，默认为 output.tif
- `--device` 指定推理时要使用的设备，可以是 CPU 或 GPU，默认为 0 卡
- `--no-show` 指定是否不显示分割图结果，默认显示
- `--no-save` 指定是否不保存分割图结果，默认保存

## <a name="license"></a> 许可证

该项目基于 [MIT 许可证](LICENSE)发行。

# 3D EfficientNet Implementation in Tensorflow 2.x

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/3DefficientNet?style=for-the-badge)](https://pypi.org/project/3DefficientNet/1.1.0/)
![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)
![CodeSize](https://img.shields.io/github/languages/code-size/armin-azh/3DefficientNet?style=for-the-badge)
[![Gmail](https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto://arminmk18@gmail.com)
![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)

This is a pure implementation of 3D effiecientnet without any pretrained weight.

you can **download** EfficientNet article from hear [arxiv](https://arxiv.org/pdf/1905.11946.pdf).


### Requirement
* tensorflow >= 2.x

### Setup
    pip install 3DefficientNet

### Sample
    from 3DefficientNet import EfficientNetB0

    model = EfficientNetB0(input_shape = (512,512,64,1), classes = 2)

### Reference
* 3D EfficientNet Torch Implementation [link](https://github.com/shijianjian/EfficientNet-PyTorch-3D).
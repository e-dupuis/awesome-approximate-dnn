# Approximate Computing in Deep Neural Networks

Hoping to give a clear view on the subject with curated contents organized

- [Approximate Computing in Deep Neural Networks](#approximate-computing-in-deep-neural-networks)
  * [Best Surveys](#best-surveys)
  * [Approximations Frameworks](#approximations-frameworks)
  * [Approximation Methods](#approximation-methods)
    + [Pruning](#pruning)
      - [Unstructured](#unstructured)
      - [Structured - Sub-kernel](#structured---sub-kernel)
      - [Structured - Kernel](#structured---kernel)
      - [Structured - filter](#structured---filter)

<small><i><a href='http://ecotrust-canada.github.io/markdown-toc/'>Table of contents generated with markdown-toc</a></i></small>


## Best Surveys

- 2019 [Deep Neural Network Approximation for Custom Hardware:Where We’ve Been, Where We’re Going](https://arxiv.org/abs/1901.06955), Wang & al.
- 2017 [Efficient Processing of Deep Neural Networks: A Tutorial and Survey](https://ieeexplore.ieee.org/document/8114708), Sze & al.
- 2020 [Model Compression and Hardware Acceleration for Neural Networks: A Comprehensive Survey](http://ieeexplore.ieee.org/document/9043731), Deng & al.
- 2020 [Approximation Computing Techniques to Accelerate CNN Based Image Processing Applications – A Survey in Hardware/Software Perspective](https://www.researchgate.net/publication/342754132_Approximation_Computing_Techniques_to_Accelerate_CNN_Based_Image_Processing_Applications_-_A_Survey_in_HardwareSoftware_Perspective), Manikandan & al.

## Approximations Frameworks

- [NEMO](https://github.com/pulp-platform/nemo) - small library for minimization of Deep Neural Networks developed in PyTorch (PTQ, QAT), support ONNX export, intended for ultra low power devices like pulp-nn 

## Dedicated HW

## Approximation Methods

### Pruning

#### Unstructured

#### Structured - Sub-kernel

#### Structured - Kernel

#### Structured - Filter

#### Structured - Hardware Friendly Structure

- [Accelerating Sparse DNN Models without Hardware-Support via Tile-Wise Sparsity](https://arxiv.org/pdf/2008.13006.pdf) - Large matrix multiplication are tiled, this method propose to maintain a regular pattern at the tile level, improving efficiency.

# Approximate Computing in Deep Neural Networks

Hoping to give a clear view on the subject with curated contents organized

- [Approximate Computing in Deep Neural Networks](#approximate-computing-in-deep-neural-networks)
  * [Lexical](#lexical)
  * [Best Surveys](#best-surveys)
  * [Tools](#tools)
    + [Approximations Frameworks](#approximations-frameworks)
    + [Dedicated Library / Compiler](#dedicated-library---compiler)
    + [Dedicated HW](#dedicated-hw)
    + [Optimization Framework](#optimization-framework)
  * [Approximation Methods](#approximation-methods)
    + [Pruning](#pruning)
      - [Unstructured](#unstructured)
      - [Structured - Sub-kernel](#structured---sub-kernel)
      - [Structured - Kernel](#structured---kernel)
      - [Structured - Filter](#structured---filter)
      - [Structured - Hardware Friendly Structure](#structured---hardware-friendly-structure)

<small><i><a href='http://ecotrust-canada.github.io/markdown-toc/'>Table of contents generated with markdown-toc</a></i></small>


## Lexical

- PQT: Post Training Quantization
- QAT: Quantization Aware Training


## Best Surveys

- 2019 [Deep Neural Network Approximation for Custom Hardware:Where We’ve Been, Where We’re Going](https://arxiv.org/abs/1901.06955), Wang & al.
- 2017 [Efficient Processing of Deep Neural Networks: A Tutorial and Survey](https://ieeexplore.ieee.org/document/8114708), Sze & al.
- 2020 [Model Compression and Hardware Acceleration for Neural Networks: A Comprehensive Survey](http://ieeexplore.ieee.org/document/9043731), Deng & al.
- 2020 [Approximation Computing Techniques to Accelerate CNN Based Image Processing Applications – A Survey in Hardware/Software Perspective](https://www.researchgate.net/publication/342754132_Approximation_Computing_Techniques_to_Accelerate_CNN_Based_Image_Processing_Applications_-_A_Survey_in_HardwareSoftware_Perspective), Manikandan & al.

## Tools

### Approximations Frameworks
| Name | Description | Framework | Supported Approx|
|---|---|---|---|
| [NEMO](https://github.com/pulp-platform/nemo) | small library for minimization of DNNs intended for ultra low power devices like pulp-nn | PyTorch, ONNX | PQT, QAT|
| [NNI](https://github.com/microsoft/nni) | lightweight toolkit for Feature Engineering, Neural Architecture Search, Hyperparameter Tuning and Model Compression | Pytorch, Tensorflow (+Keras), MXnet, Caffe2 CNTK, Theano | Pruning / PQT)|
| [PocketFlow](https://github.com/Tencent/PocketFlow) | open-source framework for compressing and accelerating DNNs. | Tensorflow | PQT, QAT, Prunning |
| [Tensorflow Model Optimization](https://github.com/tensorflow/model-optimization/) | Toolkit to optimize ML / DNN model | Tenforflow(Keras) | Clustering, Quantization (PQT, QAT), Pruning |

### Dedicated Library / Compiler

- PULP-NN [code](https://github.com/pulp-platform/pulp-nn), [paper](https://arxiv.org/abs/1908.11263) - QNN inference library for ultra low power PULP RiscV core
- [DORY](https://github.com/pulp-platform/dory) - automatic tool to deploy DNNs on low-cost MCUs with typically less than 1MB of on-chip SRAM memory
- [Glow](https://github.com/pytorch/glow) - Glow is a machine learning compiler and execution engine for hardware accelerators (Pytorch, ONNX) 

### Dedicated HW (ASIC)
| Name | Description | Environment | Perf |
|---|---|---|---|
|[Esperanto ET-soc-1](https://www.esperanto.ai/esperanto-technologies-to-reveal-chip-with-1000-cores-at-risc-v-summit/) | 1000+ low power risc v core chip energy efficient processing of ML/DNN | Cloud | 800 TOPS @ 20W |
|[Google TPU](https://cloud.google.com/tpu/docs/tpus) | Processing unit for DNN workload, efficient systolic array for computation | Cloud, Edge | V3 - 90 TOPS @250W, Coral Edge 4TOPS @ 2W |

### FPGA based accelerator

- [Maestro](https://github.com/maestro-project/maestro) - open-source tool for modeling and evaluating the performance and energy-efficiency of different dataflows for DNNs



## Approximation Methods

### Pruning

#### Unstructured

#### Structured - Sub-kernel

#### Structured - Kernel

#### Structured - Filter

#### Structured - Hardware Friendly Structure

- [Accelerating Sparse DNN Models without Hardware-Support via Tile-Wise Sparsity](https://arxiv.org/pdf/2008.13006.pdf) - Large matrix multiplication are tiled, this method propose to maintain a regular pattern at the tile level, improving efficiency.

## Others

### Optimization Framework

- [Google OR-Tools](https://developers.google.com/optimization/introduction/overview)

### Simulation Framework

- [Renode](https://github.com/renode/renode) - Simulation platform for MCU dev & test (functional, single and multi-node)

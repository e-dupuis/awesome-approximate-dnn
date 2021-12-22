# Approximate Computing in Deep Neural Networks

Hoping to give a clear view on the subject with curated contents organized

From algorithm to hardware execution

- [Approximate Computing in Deep Neural Networks](#approximate-computing-in-deep-neural-networks)
  * [Lexical](#lexical)
  * [Best Surveys](#best-surveys)
  * [Tools](#tools)
    + [Approximations Frameworks](#approximations-frameworks)
    + [Dedicated Library](#dedicated-library)
    + [Graph Compiler](#graph-compiler)
    + [Commercial Dedicated HW accelerator (ASIC)](#commercial-dedicated-hw-accelerator--asic-)
    + [FPGA based accelerator / HLS for CNNs](#fpga-based-accelerator---hls-for-cnns)
    + [Evaluation Frameworks](#evaluation-frameworks)
    + [Simulation Frameworks](#simulation-frameworks)
  * [Approximation Methods](#approximation-methods)
    + [Pruning](#pruning)
      - [Structured - Hardware Friendly Structure](#structured---hardware-friendly-structure)
      - [Weight Saliency Determination](#weight-saliency-determination)
      - [Data-free methods](#data-free-methods)
    + [Quantization](#quantization)
    + [Approximate operators](#approximate-operators)
  * [Others](#others)
    + [Contests](#contests)
    + [Model ZOO](#model-zoo)
    + [Generic DSE Framework](#generic-dse-framework)
    + [DNN conversion framework](#dnn-conversion-framework)
    + [Visualization Framework](#visualization-framework)
    + [HLS Framework](#hls-framework)

<small><i><a href='http://ecotrust-canada.github.io/markdown-toc/'>Table of contents generated with markdown-toc</a></i></small>

## Lexical

- PTQ: Post Training Quantization
- QAT: Quantization Aware Training


## Best Surveys

- 2019 [Deep Neural Network Approximation for Custom Hardware:Where We’ve Been, Where We’re Going](https://arxiv.org/abs/1901.06955), Wang & al.
- 2017 [Efficient Processing of Deep Neural Networks: A Tutorial and Survey](https://ieeexplore.ieee.org/document/8114708), Sze & al.
- 2019 [Recent Advances in Convolutional Neural Network Acceleration](https://www.semanticscholar.org/paper/Recent-Advances-in-Convolutional-Neural-Network-Zhang-Zhang/9552e625973a8c67a7e709cc4aa75c4fc71ce261), Qianru Zhang, & al.
- 2020 [Model Compression and Hardware Acceleration for Neural Networks: A Comprehensive Survey](http://ieeexplore.ieee.org/document/9043731), Deng & al.
- 2020 [Approximation Computing Techniques to Accelerate CNN Based Image Processing Applications – A Survey in Hardware/Software Perspective](https://www.researchgate.net/publication/342754132_Approximation_Computing_Techniques_to_Accelerate_CNN_Based_Image_Processing_Applications_-_A_Survey_in_HardwareSoftware_Perspective), Manikandan & al.
- 2021 [Pruning and Quantization for Deep Neural Network Acceleration: A Survey](https://arxiv.org/abs/2101.09671), Liang & al.

## Tools

### Approximations Frameworks
| Name | Description | Framework | Supported Approx|
|---|---|---|---|
| [NEMO](https://github.com/pulp-platform/nemo) | small library for minimization of DNNs intended for ultra low power devices like pulp-nn | PyTorch, ONNX | PTQ, QAT|
| [Microsoft NNI](https://github.com/microsoft/nni) | lightweight toolkit for Feature Engineering, Neural Architecture Search, Hyperparameter Tuning and Model Compression | Pytorch, Tensorflow (+Keras), MXnet, Caffe2 CNTK, Theano | Pruning / PTQ)|
| [PocketFlow](https://github.com/Tencent/PocketFlow) | open-source framework for compressing and accelerating DNNs. | Tensorflow | PTQ, QAT, Prunning |
| [Tensorflow Model Optimization](https://github.com/tensorflow/model-optimization/) | Toolkit to optimize ML / DNN model | Tenforflow(Keras) | Clustering, Quantization (PTQ, QAT), Pruning |
| [QKeras](https://github.com/google/qkeras) | quantization extension to Keras that provides drop-in replacement for some of the Keras layers| Tensorflow(Keras) | Quantization (QAT) |
| [Brevitas](https://github.com/Xilinx/brevitas) | Pytorch extension to quantize DNN model | Pytorch | PTQ, QAT | 
| [TFApprox](https://github.com/ehw-fit/tf-approximate) | Add ApproxConv layers to TF to emulate the use of approximated multipliers on GPU, typically from [EvoApproxLib](https://github.com/ehw-fit/evoapproxlib)  | Tensorflow | Approximate Multipliers|
|[N2D2](https://github.com/CEA-LIST/N2D2)| Toolset to import or train model, apply quantization, and export in various format (C/C++ ...)|ONNX|QAT(license required), PTQ|

### Dedicated Library

- PULP-NN [code](https://github.com/pulp-platform/pulp-nn), [paper](https://arxiv.org/abs/1908.11263) - QNN inference library for ultra low power PULP RiscV core

### Graph Compiler

- [DORY](https://github.com/pulp-platform/dory) - automatic tool to deploy DNNs on low-cost MCUs with typically less than 1MB of on-chip SRAM memory
- [Glow](https://github.com/pytorch/glow) - Glow is a machine learning compiler and execution engine for hardware accelerators (Pytorch, ONNX) 
- [TensorflowLite](https://www.tensorflow.org/lite/guide) - TensorFlow Lite is a set of tools to help developers run TensorFlow models on mobile, embedded, and IoT devices. It enables on-device machine learning inference with low latency and a small binary size (linux, android, mcu). [curated content for tflite](https://github.com/margaretmz/awesome-tensorflow-lite)
- [OpenVino](https://docs.openvinotoolkit.org) - OpenCL based graph compiler for intel environnment (Intel CPU, Intel GPU, Dedicated accelerator)
- [N2D2](https://github.com/CEA-LIST/N2D2) - Framework capable of training and exporting DNN in different format, particulary standalone C/C++ compilable project with very few dependencis and quantized, support import from ONNX model
- [Vitis AI](https://www.xilinx.com/products/design-tools/vitis/vitis-ai.html) - Optimal Artificial Intelligence Inference from Edge to Cloud (compiler / optimizer / quantizer / profiler / IP set)

### Commercial Dedicated HW accelerator (ASIC)
| Name | Description | Environment | Perf |
|---|---|---|---|
|[Esperanto ET-soc-1](https://www.esperanto.ai/esperanto-technologies-to-reveal-chip-with-1000-cores-at-risc-v-summit/) | 1000+ low power risc v core chip energy efficient processing of ML/DNN | Cloud | 800 TOPS @ 20W |
|[Google TPU](https://cloud.google.com/tpu/docs/tpus) | Processing unit for DNN workload, efficient systolic array for computation | Cloud, Edge | V3 - 90 TOPS @250W, Coral Edge 4TOPS @ 2W |
|[Greenwave GAP8](https://ieeexplore.ieee.org/document/8445101)| multi-GOPS fully programmable RISC-V IoT-edge computing engine, featuring a 8-core cluster with CNN accelerator, coupled with an ultra-low power MCU with 30 μW state-retentive sleep power (75mW)|Edge| 600 GMAC/s/W|
|[Intel Movidius Myriad](https://www.intel.com/content/www/us/en/products/docs/processors/movidius-vpu/myriad-x-product-brief.html)| Vector processing unit for accelerating DNN inference, Interface with the OpenVino toolkit, 16 programmable cores| Edge | 1 TOPS @ 1.5W - 2.67 TOPS/W|

### FPGA based accelerator / HLS for CNNs

- [Maestro](https://github.com/maestro-project/maestro) - open-source tool for modeling and evaluating the performance and energy-efficiency of different dataflows for DNNs
- [HLS4ML](https://github.com/fastmachinelearning/hls4ml) - package for creating HLS from various ML framework (good pytorch support), create streamline architecture
- [FINN](https://github.com/Xilinx/finn) - framework for creating HW accelerator (HLS code) from BREVITAS quantized model, downto BNN, create PE architecture
- [N2D2](https://github.com/CEA-LIST/N2D2) - framework for creating HLS from N2D2 trained model (support ONNX import), create streamline architecture
- [ScaleHLS](https://github.com/hanchenye/scalehls) - HLS framework on MLIR. Can compile HLS C/C++ or ONNX model to optimized HLS C/C++ in order to generate high-efficiency RTL design using downstream tools, such as Vivado HLS. Focus on scalability, automated DSE engine.

### Evaluation Frameworks

- [DNN-Neurosim](https://github.com/neurosim/DNN_NeuroSim_V2.0) - Framework for evaluating the performance of inference or training of on-chip DNN

### Simulation Frameworks
- [SCALE-Sim](https://github.com/ARM-software/SCALE-Sim) - ARM CNN accelerator simulator, that provides cycle-accurate timing, power/energy, memory bandwidth and trace results for a specified accelerator configuration and neural network architecture.
- [Eyeriss Energy Estimator](https://energyestimation.mit.edu) - Energy Estimator for MIT's Eyeriss Hardware Accelerator
- [Torchbench](https://github.com/paperswithcode/torchbench) - collection of deep learning benchmarks you can use to benchmark your models, optimized for the PyTorch framework.
- [Renode](https://github.com/renode/renode) - Functional simulation platform for MCU dev & test (single and multi-node)

## Approximation Methods

- 2018 [Awesome-model-compression-and-acceleration](https://github.com/memoiry/Awesome-model-compression-and-acceleration) - Github rebo with list of compression and acceleration techniques for Deep Learning 
- 2019 [Model Compression and Acceleration Progress](https://github.com/juliagusak/model-compression-and-acceleration-progress) -  Github rebo with list of compression and acceleration techniques for Deep Learning 

### Multi-techniques

- 2020 [Deep Neural Network Compression by In-Parallel Pruning-Quantization](https://ieeexplore.ieee.org/document/8573867) - Use Bayesian optimization to solve both pruning and quantization problems jointly and with fine-tuning.

- 2020 [OPQ: Compressing Deep Neural Networks with One-shot Pruning-Quantization](https://www.semanticscholar.org/paper/OPQ%3A-Compressing-Deep-Neural-Networks-with-One-shot-Hu-Peng/7b16367b575d951a98f1762d8f45d7c0eb840581) - Analytical single shot compression (Pruning + Quantization) of DNN using only pretrained weights values, then fine-tuning to recover ACL 

### Pruning

#### Structured - Hardware Friendly Structure

- [Accelerating Sparse DNN Models without Hardware-Support via Tile-Wise Sparsity](https://arxiv.org/pdf/2008.13006.pdf) - Large matrix multiplication are tiled, this method propose to maintain a regular pattern at the tile level, improving efficiency.

#### Weight Saliency Determination

- 2020 [Utilizing Explainable AI for Quantization and Pruning of Deep Neural Networks](https://arxiv.org/abs/2008.09072) - Using DeepLift (explainable AI) as hints to improve compression by determining importance of neurons and features

#### Data-free methods
- 2021 [Post-training deep neural network pruning via layer-wise calibration](https://arxiv.org/abs/2104.15023) - Layer-wise sparse pruning calibration based on the use of fractal images to replace representative data, post quantization, achieving 2x compression.

### Quantization

- 2018 [Learning Compression from Limited Unlabeled Data](https://openaccess.thecvf.com/content_ECCV_2018/papers/Xiangyu_He_Learning_Compression_from_ECCV_2018_paper.pdf) - Use unlabelled data to improve accuracy of quantization in a very fast fine-tuning step
- 2020 [Automatic heterogeneous quantization of deep neural networks for low-latency inference on the edge for particle detectors](https://arxiv.org/pdf/2.006.10159.pdf) - AutoQKeras, Per layer quantization optimization using meta-heuristic DSE based on Bayesian Optimization, make use of Qkeras & hls4ml.

### Approximate operators
- 2020 [Full Approximation of Deep Neural Networks through Efficient Optimization](https://ieeexplore.ieee.org/document/9181236/) - Select efficient approx multipliers through retraining and minimization of accuracy loss (Evo Approx)
- 2019 [ALWANN: Automatic Layer-Wise Approximation of Deep Neural Network Accelerators without Retraining](https://arxiv.org/abs/1907.07229) - Use NSGA II to optimize approximate multipliers implemented & DNN mapping onto implemented Ax multipliers (Evo Approx).


## Others

### Contests

- [MLPerf / MLCommons](https://mlcommons.org/en/) - Acceleration contest for ML
- [Papers with Code](https://paperswithcode.com) - latest papers / code in ML, SoTA representation for several applications (CV, NLP, Medical ...)

### Model ZOO

- [TIMM](https://github.com/rwightman/pytorch-image-models) - Excellent model zoo & training scripts for pytorch
- [ONNX Model Zoo](https://github.com/onnx/models) - Collection of pre-trained onnx models
- [Tensorflow Hub](https://tfhub.dev) - pre-trained model that can be imported as keras layers for deployment / fine-tuning
- [Keras Applications](https://www.tensorflow.org/api_docs/python/tf/keras/applications) - pre-trained popular CNNs implemented in Keras - can be customized and fine tuned
- [Torchvision](https://pytorch.org/vision/stable/models.html) - The torch equivalent to keras applications
- [Openvino pre-trained models](https://docs.openvino.ai/2019_R1/_docs_Pre_Trained_Models.html) - Intel pre-trained model for use in OpenVino

### Generic DSE Framework

- [Google OR-Tools](https://developers.google.com/optimization/introduction/overview) - Constraint programming, routing and other optimization tools
- [Facebook Botorch](https://botorch.org) - Bayesian optimization accelerated by torch backend, python API
- [Pymoo](https://botorch.org) - collection of multi-objective optimization implementation in python, user friendly interface

### DNN conversion framework

- [MMdnn](https://github.com/Microsoft/MMdnn) - Microsoft tool for cross-framework conversion, retraining, visualization & deployment
- [ONNX](https://github.com/onnx/onnx) - model format to exchange frozen models between ML frameworks 

### Visualization Framework

- [Tensorboard](https://www.tensorflow.org/tensorboard) - Visualization tool for Tensorflow, Pytorch ..., can show graph, metric evolution over training ... very adaptable
- [Netron](https://github.com/lutzroeder/netron) - Tool to show ONNX graph with all the attributes.
- [mlflow](https://mlflow.org) - very flexible simulation logging tool (client/server) allowing to log parameter & metrics + object storage, python and shell interfaces 

### HLS Framework

- [Xilinx Vivado HLS](https://www.xilinx.com/support/documentation-navigation/design-hubs/dh0012-vivado-high-level-synthesis-hub.html) - C/C++ based HLS for XILINX Fpga
- [ntel Quartus HLS](https://www.intel.fr/content/www/fr/fr/software/programmable/quartus-prime/hls-compiler.html) - C++ HLS for ALTERA/INTEL FPGA
- [Mentor Catapult HLS](https://eda.sw.siemens.com/en-US/ic/ic-design/high-level-synthesis-and-verification-platform/) - C++/SystemC HLS For Siemens FPGA

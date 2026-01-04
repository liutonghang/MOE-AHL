# Learning Language-guided Adaptive Hyper-modality Representation for Multimodal Sentiment Analysis

Pytorch implementation of paper: 

> [**基于MOE-AHL架构的多模态情感识别方法研究**](广西大学刘通航的研究生毕业论文 2016-01-04)

> This is a reorganized code, if you find any bugs please contact me. Thanks.


## Content
- [Note](#Note)
- [Data Preparation](#Data-preparation)
- [Environment](#Environment)
- [Training](#Training)
- [Citation](#Citation)


## Note

1. [2026.01.04] The demo code has been updated to fix some issues. We recommend reproducing with new code and environmental requirements.

2. Based on the experience and insights gained from the MOE-AHL, we have futher explored robust MSA by ensuring the integrity of the dominant modality under different noise intensities.

3. The MOE_AHL implementation has not been added to [MMSA](https://github.com/thuiar/MMSA); but you can also refer to the implementation and make a fairer comparison with other methods in the same framework.

4. We observed that regression metrics (such as MAE and Corr) and classification metrics (such as acc2 and F1) focus on different aspects of model performance. A model that achieves the lowest error in sentiment intensity prediction does not necessarily perform best in classification tasks. To comprehensively demonstrate the capabilities of the model, we selected the best-performing model for each type of metric, meaning that acc2/F1 and MAE correspond to different epochs of the same training process. In addition, the code also compute and report the performance in the same epoch for reference.


## Data Preparation
MOSI/MOSEI/CH-SIMS Download: See [MMSA](https://github.com/thuiar/MMSA).

## Environment
The basic training environment for the results in the paper is Pytorch 2.4.1 with CUDA 11.8, Python 3.9.23 with RTX A40.
It should be noted that different hardware and software environments can cause the results to fluctuate.

## Training
You can quickly run the code with the following command:

### CH-SIMS
```
python train.py --config_file configs/sims.yaml
```

### MOSI
```
python train.py --config_file configs/mosi.yaml
```

### MOSEI
```
python train.py --config_file configs/mosei.yaml
```

## ablation experiments 
You can quickly run the code with the following command:


### CH-SIMS
```
python ablation_train.py --config_file ablation_config/sims.yaml
```

### MOSI
```
python ablation_train.py --config_file ablation_config/mosi.yaml
```

### MOSEI
```
python ablation_train.py --config_file ablation_config/mosei.yaml
```

## Contact me
Any comments and suggestions are welcome. Please feel free to contact the authors via email 377420029@qq.com . Thank you!


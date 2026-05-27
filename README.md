Official implementation for the ICCV 2025 paper:

**Robust Machine Unlearning for Quantized Neural Networks via Adaptive Gradient Reweighting with Similar Labels**  
Yujia Tong, Yuze Wang, Jingling Yuan, Chuang Hu  
*Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2025*

## Overview

This repository provides the official code for our paper on robust machine unlearning for quantized neural networks. The proposed method addresses the challenge of removing the influence of specific training data from quantized models while preserving model utility and robustness.

Our approach introduces **Adaptive Gradient Reweighting with Similar Labels**, which improves unlearning effectiveness by adaptively adjusting gradient contributions from samples with semantically or label-wise similar classes. This design helps quantized neural networks forget targeted data more reliably while reducing performance degradation on retained data.

## Method

Machine unlearning aims to make a trained model behave as if certain data had never been used during training. This is especially challenging for quantized neural networks because quantization can amplify optimization instability and make fine-grained parameter updates harder.

The proposed framework focuses on:

- Robust unlearning for quantized neural networks
- Adaptive gradient reweighting during the unlearning process
- Use of similar-label information to guide forgetting
- Better balance between forgetting efficacy and retained accuracy
- Compatibility with low-precision neural network deployment

## Citation

If you find this repository useful for your research, please cite our paper:

```bibtex
@InProceedings{Tong_2025_ICCV,
    author    = {Tong, Yujia and Wang, Yuze and Yuan, Jingling and Hu, Chuang},
    title     = {Robust Machine Unlearning for Quantized Neural Networks via Adaptive Gradient Reweighting with Similar Labels},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2025},
    pages     = {20603-20612}
}

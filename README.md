# Robust Machine Unlearning for Quantized Neural Networks via Adaptive Gradient Reweighting with Similar Labels

## Overview

This repository provides the official implementation of our ICCV 2025 paper:

**Robust Machine Unlearning for Quantized Neural Networks via Adaptive Gradient Reweighting with Similar Labels**  
Yujia Tong, Yuze Wang, Jingling Yuan, Chuang Hu  
*Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2025*

Machine unlearning aims to remove the influence of specified training data from a trained model while preserving its performance on the remaining data. This task becomes more challenging for quantized neural networks, where low-precision weights and activations may introduce additional optimization instability during the unlearning process.

To address this problem, we propose **Adaptive Gradient Reweighting with Similar Labels**, a robust machine unlearning method designed for quantized neural networks. The method adaptively adjusts gradient contributions using similar-label information, improving forgetting effectiveness while maintaining the utility of the quantized model.

## News

- **May 2026**: Our follow-up work, **[Forget by Uncertainty: Orthogonal Entropy Unlearning for Quantized Neural Networks](https://arxiv.org/abs/2602.00567)**, has been accepted by **ICML 2026**.


## Usage

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Original Model

Train the original model before performing machine unlearning:

```bash
python main_train.py \
    --arch {model name} \
    --dataset {dataset name} \
    --epochs {epochs for training} \
    --lr {learning rate for training} \
    --save_dir {file to save the origin model}
```


### 3. Perform Unlearning

Run the unlearning procedure with the trained original model:

```bash
python main_forget.py \
    --save_dir ${save_dir} \
    --model_path ${origin_model_path} \
    --unlearn RL_min \
    --num_indexes_to_replace ${forgetting data amount} \
    --unlearn_epochs ${epochs for unlearning} \
    --unlearn_lr ${learning rate for unlearning}
```



## Follow-up Work

Our follow-up work further studies machine unlearning for quantized neural networks from the perspective of uncertainty-based forgetting:

**Forget by Uncertainty: Orthogonal Entropy Unlearning for Quantized Neural Networks**  
Tian Zhang, Yujia Tong, Junhao Dong, Ke Xu, Yuze Wang, Jingling Yuan  
Accepted by **ICML 2026**  
Paper: [https://arxiv.org/abs/2602.00567](https://arxiv.org/abs/2602.00567)



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
```

## Acknowledgements

We sincerely thank the authors of [OPTML-Group/Unlearn-Saliency](https://github.com/OPTML-Group/Unlearn-Saliency) for releasing their codebase and contributing to the machine unlearning community.

# 🚀 AdaSFormer: Adaptive Serialized Transformer for Indoor Monocular Semantic Scene Completion


## 🎉 News

- 🔥 Accepted by CVPR 2026 (Feb 2026)
- 🚀 Code released in March 2026

## 📌 Overview
Indoor Monocular Semantic Scene Completion (MSSC) aims to recover a complete 3D semantic scene from a single RGB image. Compared to outdoor settings, indoor scenes exhibit complex spatial layouts, high object density, and severe occlusions, making the task significantly more challenging.

To address these challenges, we propose **AdaSFormer**, a **serialized transformer framework** tailored for indoor MSSC. Our method effectively balances global context modeling and local detail reconstruction through a hybrid design of transformers and convolutional modules.

---

## ✨ Key Features
AdaSFormer introduces three core innovations:

- **Adaptive Serialized Attention (ASA)**  
  Dynamically adjusts receptive fields via learnable shifts, enabling flexible long-range dependency modeling.

- **Center-Relative Positional Encoding (CRPE)**  
  Encodes spatial information richness relative to the scene center, improving reasoning in occluded regions.

- **Convolution-Modulated Layer Normalization (CMLN)**  
  Bridges the representation gap between convolutional and transformer features for better feature alignment.

---

## 🧠 Framework
Given a single RGB image:

1. A **2D encoder** extracts image features and predicts depth  
2. Features are projected into 3D space via surface projection  
3. A **3D encoder** alternates between:
   - Transformer modules → capture global context  
   - Convolution modules → refine local geometry  
4. A lightweight decoder produces final semantic scene completion results

![Framework Illustration](./figures/framework.png)  <!-- 可替换为你的结构图路径 -->

---

## 🏆 Results
AdaSFormer achieves **state-of-the-art performance** on:

- **NYUv2**
- **Occ-ScanNet**

while maintaining **lower memory consumption** and **higher efficiency** compared to existing transformer-based methods.

---

## 📖 Citation

If you find our work useful, please consider citing:

```bibtex
@inproceedings{AdaSFormer2026_cvpr,
  author    = {Xuzhi Wang and Xinran Wu and Song Wang and Lingdong Kong and Ziping Zhao},
  title     = {AdaSFormer: Adaptive Serialized Transformers for Monocular Semantic Scene Completion from Indoor Environments},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2026}
}

@inproceedings{AdaSFormer2026_arxiv,
  author  = {Xuzhi Wang and Xinran Wu and Song Wang and Lingdong Kong and Ziping Zhao},
  title   = {AdaSFormer: Adaptive Serialized Transformers for Monocular Semantic Scene Completion from Indoor Environments},
  booktitle = {arXiv preprint arXiv:2603.25494},
  year    = {2026}
}


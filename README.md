# LanSE: Language-Grounded Sparse Encoders

[![arXiv](https://img.shields.io/badge/arXiv-2508.18236-b31b1b.svg)](https://arxiv.org/pdf/2508.18236)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

Official implementation of **"Human-like Content Analysis for Generative AI with Language-Grounded Sparse Encoders"**

> **Authors:** Yiming Tang¹*, Arash Lagzian¹, Srinivas Anumasa¹, Qiran Zou¹, Yingtao Zhu¹'², Ye Zhang¹'³, Trang Nguyen⁴, Yih-Chung Tham¹, Ehsan Adeli⁴, Ching-Yu Cheng¹, Yilun Du⁵, Dianbo Liu¹*
>
> ¹National University of Singapore, ²Tsinghua University, ³Capital Medical University, ⁴Stanford University, ⁵Harvard University
>
> *Corresponding authors

---

## 🔍 Overview

LanSE provides **systematic, interpretable content analysis** for generative AI by decomposing images into thousands of natural language-described visual patterns. Unlike existing holistic metrics (FID, CLIP, IS), LanSE enables fine-grained, human-understandable evaluation of AI-generated content.

### Key Features

- **5,309 interpretable visual patterns** with 93% human agreement on natural images
- **899 clinical patterns** for medical imaging (74% radiologist agreement)
- **Four diagnostic metrics**: Prompt Match, Visual Realism, Physical Plausibility, Content Diversity
- **Cross-domain applicability**: Natural images and chest X-rays
- **Outperforms LMMs**: 76.6% vs 72.3% (GPT-4o) on physics violation detection

### Why LanSE?

Traditional evaluation methods treat images as indivisible wholes, while real-world AI failures manifest as **specific visual patterns**. LanSE bridges this gap by:

1. **Discovering** interpretable visual patterns using sparse autoencoders
2. **Describing** patterns in natural language via large multimodal models
3. **Evaluating** content through pattern-based decomposition

---

## 📦 Installation

### Prerequisites

- Python 3.9 or higher
- CUDA 11.8+ (for GPU acceleration)
- 16GB+ RAM recommended

### Setup
```bash

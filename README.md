# Deep-Learning-and-applied-AI

# Correcting Shortcut Learning via Model Merging on CelebA

**Deep Learning and Applied AI 2025/2026 — Sapienza University of Rome**  
Anaïs Augé & Manal Ghorafi

## Overview

This project investigates whether **model merging** can correct shortcut learning in a biased classifier without retraining. On CelebA, a ResNet-50 trained with standard ERM exploits the gender–blondness correlation, reaching 94.6% average accuracy but only 22.0% worst-group accuracy (WGA) on blond males.

We train a lightweight specialist exclusively on male subgroups and merge it with the ERM model using four strategies. Our best method, **Task Arithmetic, achieves 84.7% WGA — surpassing Group DRO (84.1%) at zero retraining cost**.

## Methods

| Method | Type | WGA | Avg Acc |
|--------|------|-----|---------|
| ERM | Baseline | 22.0% | 94.6% |
| UW | Retrained | 83.5% | 88.9% |
| Group DRO | Retrained | 84.1% | 90.5% |
| Naive Merge | Merge | 80.2% | 88.2% |
| **Task Arithmetic** | **Merge** | **84.7%** | **89.0%** |
| Layer-Selective TA | Merge | 84.6% | 90.0% |
| Fisher Merging | Merge | 64.8% | 91.4% |
| Ensemble Logit | Output-space | 80.8% | 88.3% |

## Requirements

```bash
pip install torch torchvision datasets scikit-learn matplotlib

# LLM–GAN Hybrid Framework for Fake Review Detection


## Project Overview

This dissertation project proposes and implements a hybrid LLM–GAN framework for automated fake review detection, operating exclusively on **textual review content**, no metadata such as ratings, user profiles or timestamps is used. This design choice ensures the framework remains broadly applicable across platforms where metadata may be unavailable, incomplete or intentionally manipulated.

The framework combines:
- **Deep semantic understanding** from pretrained BERT (bert-base-uncased) embeddings
- **Explicit aspect-opinion signals** from TF-IDF encoding of preprocessed review text
- **Adversarial learning** through a dual discriminator GAN architecture inspired by FakeGAN

---

## Architecture Overview

The framework follows five main stages:

1. **Text Preprocessing** — HTML removal, lowercasing, tokenisation, stopword removal
2. **Aspect & Opinion Extraction** — POS tagging to extract nouns, noun phrases, adjectives and adverbs, stored separately from the original review text
3. **Dual-Input Embedding Generation**
   - Full review text → BERT (bert-base-uncased, max 256 tokens) → 768-dim CLS token embedding
   - Preprocessed aspect-opinion text → TF-IDF (5,000 features) → TruncatedSVD (50 dims)
   - Concatenated into a single **818-dimensional combined vector** per review
4. **FakeGAN-Inspired Dual Discriminator Adversarial Training**
   - Generator G: produces synthetic deceptive embeddings conditioned on random noise and real review embeddings
   - Discriminator D: primary classifier — genuine vs fake
   - Discriminator D': secondary — distinguishes real deceptive from generator-produced embeddings
5. **Inference** — Generator disabled; D alone classifies reviews via sigmoid + threshold 0.5

---

## Implementation Details

### Pre-Training Strategy (Sequential)
All three components are pre-trained before adversarial competition begins:

| Component | Loss | Notes |
|---|---|---|
| Discriminator D | BCE with logits | Supervised classifier on labelled real/fake reviews |
| Generator G | MSE Loss, Cosine Similarity Loss, Statistical Loss and Diversity Loss | Trained on deceptive embeddings only |
| Discriminator D' | BCE with logits | Distinguishes real deceptive from generated samples |

Pre-training epochs were adapted per dataset: 10 epochs (Dataset 3), 20 epochs (Dataset 2), 30 epochs (Dataset 1).

### Adversarial Training
- Update ratio: **g=1, d=4** (following FakeGAN paper)
- Optimiser: Adam, betas=(0.5, 0.999)
- Generator LR: 0.00005 | Discriminator LR: 0.00001
- Early stopping with patience of 10
- Best discriminator checkpoint based on validation balanced accuracy was restored for final evaluation

### GAN Architecture
- Generator: Linear(918→512) → LayerNorm → ReLU → Linear(512→768) → LayerNorm → ReLU → Linear(768→512) → LayerNorm → ReLU → Linear(512→818)
- Discriminator D & D': Linear(818→512) → LeakyReLU(0.2) → Dropout(0.3) → Linear(512→256) → LeakyReLU(0.2) → Dropout(0.3) → Linear(256→1)
- No sigmoid during training (raw logits) — sigmoid applied post-hoc at inference only

---

## Datasets

Three publicly available datasets were used, evaluated independently:

| Dataset | Source | Size | Type | Balance |
|---|---|---|---|---|
| Deceptive Reviews | Mendeley | ~21,000 reviews | Restaurant reviews, Google Maps NYC | Imbalanced |
| Deceptive Opinion Spam Corpus | Ott et al. / GitHub | 1,600 reviews | Hotel reviews, Mechanical Turk fakes | Balanced (50/50) |
| Maxwell Fake Reviews | Kaggle | ~40,000 reviews | E-commerce, AI-generated fakes (CG/OR) | Balanced (50/50) |

Each dataset was split 80/10/10 (train/val/test) independently. Fixed random seed 42 applied throughout for reproducibility.

---

## Results

| Metric | Dataset 1 (Mendeley) | Dataset 2 (Ott et al.) | Dataset 3 (Maxwell) |
|---|---|---|---|
| Accuracy | 0.6909 | 0.8250 | 0.9411 |
| Balanced Accuracy | 0.6819 | 0.8250 | 0.9411 |
| Precision | 0.6257 | 0.8250 | 0.9464 |
| Recall | 0.6299 | 0.8250 | 0.9352 |
| F1-Score | 0.6278 | 0.8250 | 0.9408 |
| AUC | 0.7541 | 0.9073 | 0.9883 |
| G-Mean | 0.6799 | 0.8250 | 0.9411 |

**Dataset 3** achieves 94.11% accuracy and 98.83% AUC, substantially exceeding FakeGAN's reported 89.1% on a single-domain dataset and competitive with state-of-the-art graph-based hybrid methods on AI/Computer generated review detection.

**Dataset 2** achieves 82.50% accuracy and 90.73% AUC on the gold-standard Ott et al. corpus with only 1,280 training samples — constrained by data scarcity and the inherent difficulty of detecting human-authored deceptive content.

**Dataset 1** achieves 69.09% accuracy, limited by heuristic labelling methodology rather than model capacity — evidenced by pre-training plateauing at ~67% accuracy despite 30 epochs, with adversarial training producing zero improvement over the pre-training baseline.

---

## Key Findings

- **Label quality matters as much as architecture** — Dataset 1 demonstrates that adversarial training cannot compensate for noisy, heuristically derived labels
- **The framework is most effective on AI/Computer generated fake reviews** — consistent with the growing real-world threat from LLM-generated deceptive content
- **Dual discriminator design helps mitigate mode collapse** — supported visually via PCA projection of generated vs real embeddings in the embedding space
- **Adversarial training contributes more to training stability and representation refinement than raw accuracy gains** — best reflected in AUC scores across Datasets 2 and 3
- The framework operates without metadata, fine-tuned LLMs or domain-specific features, making it broadly applicable across platforms

---

## Tech Stack

- Python 3 / Google Colab (NVIDIA T4 GPU)
- PyTorch
- HuggingFace Transformers (bert-base-uncased)
- scikit-learn (TF-IDF, TruncatedSVD, evaluation metrics)
- NumPy, Pandas, Matplotlib, Seaborn

---

## Ethical Considerations

- The generator is used exclusively for training augmentation and is not publicly released
- No synthetic reviews are deployed — only the trained discriminator is shared
- The framework minimises false positives to avoid unfair penalties on legitimate users or businesses
- Dataset biases and potential residual label uncertainty within the CG/OR dataset construction are acknowledged in the evaluation

---

## Future Work

- Evaluate on additional datasets for stronger cross-domain generalisation evidence
- Implement SHAP or attention visualisation for model explainability
- Conduct ablation studies to quantify individual BERT and TF-IDF contributions
- Explore stronger embedding backbones such as RoBERTa or ModernBERT
- Investigate threshold tuning on validation set to optimise precision-recall balance for deployment

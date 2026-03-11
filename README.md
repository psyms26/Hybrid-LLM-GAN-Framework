# LLM–GAN Hybrid Framework for Fake Review Detection

**Status:** Data preprocessing in progress. Implementation to begin upon supervisor approval.

## Project Overview
This dissertation project develops a hybrid LLM–GAN framework to detect fake reviews using **textual content only** — no metadata (ratings, user profiles, timestamps, etc.).

The approach combines:
- **Deep semantic understanding** from pre-trained LLMs (BERT/RoBERTa)
- **Adversarial robustness** from GAN training

## Key Components
- **Preprocessing**  
  HTML removal, lowercasing, tokenization, stopword removal, lemmatization.

- **Aspect & Opinion Extraction**  
  POS tagging → nouns/noun phrases (aspects) + adjectives/adverbs (opinions/intensity).  
  Recombined into concise sequences for LLM input.

- **Contextual Embeddings**  
  BERT/RoBERTa pooled embeddings ([CLS] or mean-pooling) in a shared space for generator and discriminator.

- **GAN-Based Adversarial Learning**  
  Generator creates synthetic review embeddings.  
  Discriminator classifies real vs. synthetic.  
  Adversarial training enhances robustness.

- **Evaluation**  
  Quantitative: Accuracy, Precision, Recall, F1-score, ROC-AUC.  
  Qualitative: Coherence/diversity of generated samples, interpretability of decisions.

## Current Progress
- Completed: Literature review, methodology design, system architecture, dataset selection, preprocessing pipeline.
- Ongoing: Data preprocessing & token extraction for embedding generation.
- Planned: LLM embedding layer, GAN implementation, training, evaluation, interpretability analysis.

## Datasets (text & labels only)
- Deceptive restaurant reviews (Google Maps, NYC) – labeled via deception indicators.
- Deceptive Opinion Spam Corpus (Ott et al., 2011) – hotel reviews, truthful vs. deceptive.
- Amazon product reviews (Kaggle) – 20k authentic + 20k GPT-2 generated fakes (OR/CG → 0/1).

## Ethical Considerations
- Generator used only for training — no public release or synthetic review deployment.
- Focus on minimizing false positives to avoid unfair penalties on genuine users.
- Commitment to bias reduction and compliance with data protection standards.

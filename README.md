# The Chakma Project

A Fine-Tuning Approach to Endangered Language Preservation and Word-Level Translation

**UCL MSc Data Science — COMP0087 Statistical Natural Language Processing**

## Overview

This project presents the first Chakma–English word-level translation system. We fine-tuned two open-weight large language models — LLaMA 3.1 8B and Gemma 3 4B — using QLoRA on a manually curated dataset of 20,206 Chakma–English word pairs extracted from a physical dictionary (Chakma, 1993) and verified by native speakers.

## Repository Structure
├── train_dictionary_lora.py       # LLaMA fine-tuning script
├── gemma_train.py                 # Gemma fine-tuning script
├── test_adapter.py                # Inference/evaluation script
├── Final_Chakma.csv               # Training dataset (20,206 word pairs)
├── ChakmaBridge Verified Version.csv  # Native-speaker verified dictionary
├── eval epoch 2/                  # LLaMA evaluation results (2 epochs)
├── eval llama/                    # LLaMA evaluation outputs
├── gemma eval 1 epoch/            # Gemma evaluation results
├── llama31_dict_lora/             # LLaMA LoRA adapter weights
├── chemini1_0_FV/                 # Gemma LoRA adapter weights
├── run_scripts/                   # Utility scripts
└── sentence_eval_results.csv      # Sentence-level evaluation results

## Models

| Model | Base | Method | chrF |
|-------|------|--------|------|
| Gemma-QLoRA | Gemma 3 4B | QLoRA (4-bit) | 11.10 |
| LLaMA-QLoRA | LLaMA 3.1 8B | QLoRA (4-bit) | 14.83 |

## Training

Both models were fine-tuned for 2 epochs with QLoRA (alpha=32, r=16, dropout=0.05, 4-bit quantisation) on an NVIDIA A100 GPU. Learning rate: 1e-4, batch size: 1, max token length: 256.

## Data

- **Dictionary dataset**: 20,206 romanised Chakma–English word pairs manually extracted from *Chakma Dictionary* (Pulin Bayan Chakma, 1993) and verified by three native speakers
- **MELD dataset**: ~800 Chakma–English sentence pairs (reviewed and corrected, not used in training)

## Acknowledgements

We thank Phonebuson Chakma, Pankaj Chakma, and Soumik Chakma for introducing us to linguistic resources, providing feedback on our datasets, and verifying translations.

# Generative Adapters for Gemma 3

This repository contains training and evaluation code for a multi-task, adapter-based fine-tuning setup around Gemma 3, with knowledge distillation (KD) support.

## What this project does

- Fine-tunes Gemma 3 using adapter modules instead of fully updating the full backbone.
- Supports multi-task training and evaluation with task-aware batching.
- Includes a custom trainer for combining hard-label cross-entropy with teacher-student KD loss.
- Provides configuration files and scripts for common training runs.

## Repository layout

- `adapters/`: adapter definitions and controller logic
- `data/`: task definitions, preprocessing, and batching utilities
- `metrics/`: evaluation metric helpers
- `third_party/`: model and trainer implementations adapted for this project
- `scripts/`: shell entry points for training runs
- `configs/`: JSON configuration files for training experiments

## Active task set

This repo’s main experiment currently uses these GEM tasks:

- `dart`
- `e2e_nlg`
- `squad_v2`

Other task definitions exist in `data/tasks_Gem.py` for future extension, but they are not part of the current active workflow.

## Setup

1. Create a Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Review the provided training scripts in `scripts/` and update paths or config files as needed.

## Training example

A typical training entry point is:

```bash
python finetuning_gemma3_trainer.py --config configs/gemma3_kd.json
```

## Acknowledgments

This project is built upon [Hyperformer](https://github.com/rabeehk/hyperformer) by Rabeeh Karimi Mahabadi.

# Nepali Crop Disease Diagnosis (CSC60904 Deep Learning, Group Assignment, Track T2)

A supervised deep learning system that diagnoses crop disease from a photo of a leaf, built for smallholder farmers in Nepal's Terai and mid-hill regions. Two models are trained and compared on the same dataset: a from-scratch CNN classifier (baseline) and a LoRA fine-tuned SmolVLM-256M vision-language model (proposed), which answers an open-ended question about the crop and its condition in natural language instead of predicting a fixed label.

Built for the "AI for the Himalayas 2026" hackathon scenario, Taylor's University, CSC60904 Deep Learning, May 2026 semester.

## Team

| Name | Role |
|---|---|
| Sushant Maharjan | Team Lead / Project Manager |
| Siddhant Bhattarai | Data Engineer |
| Pratham Shrestha | ML Engineer (Modelling) |
| Samyak Upadhyay | Evaluation & Ethics Lead |

## Problem & Context

Crop disease threatens food security and household income for Nepal's smallholder farmers, many of whom lack timely access to agricultural extension officers. This project explores whether a lightweight, deployable deep learning system could help close that gap by diagnosing disease directly from a smartphone photo. See the full report for the complete Nepal-context discussion, ethical analysis, and deployment considerations.

## Repository Structure

Folders are organized by **pipeline stage / ownership**, not strictly by file type:

```
data/         Data engineering: corpus-building notebook, dataset extraction script, dataset source link
notebooks/    Model training: CNN baseline, SmolVLM + LoRA fine-tuning
models/       Saved model checkpoints
results/      Evaluation: metrics, confusion matrices, evaluation notebook, class index
docs/         Project management evidence: Gantt chart, Jira board, meeting minutes
demo/         Demo interface: Gradio app for interactive crop disease diagnosis
src/          (reserved for shared/reusable source modules)
```

- `data/crop_disease_corpus_builder.ipynb`: merges and labels the three source datasets, builds train/val/test splits
- `data/extract_data.py`: downloads and extracts the prepared dataset from Hugging Face
- `data/DATASET_LINK.md`: pointer to the hosted dataset (too large to commit directly)
- `notebooks/CNN-baseline.ipynb`: trains the from-scratch baseline CNN
- `notebooks/smolvlm256m_nepali_crop_finetune.ipynb`: LoRA fine-tunes SmolVLM-256M-Instruct
- `models/baseline_cnn_final/baseline_cnn.pt`: trained CNN checkpoint
- `results/crop_disease_model_evaluations.ipynb`: loads both trained models, computes metrics, confusion matrices, error analysis
- `results/class_index.json`: class label index used by the CNN
- `demo/demo.ipynb`: Gradio interface, upload a leaf photo and see both models' predictions side by side

## Dataset

Source: three merged public datasets covering crop and vegetable diseases relevant to South Asian agriculture (no large-scale, field-collected Nepali dataset currently exists publicly):

- Kaggle: [Crop Disease Detection Dataset](https://www.kaggle.com/datasets/snikhilrao/crop-disease-detection-dataset) (snikhilrao)
- GitHub: [PlantDoc-Dataset](https://github.com/pratikkayal/PlantDoc-Dataset) (pratikkayal)
- Hugging Face: [bd-crop-vegetable-plant-disease-dataset](https://huggingface.co/datasets/Saon110/bd-crop-vegetable-plant-disease-dataset) (Saon110)

Merged, cleaned, and split corpus hosted at: **`w4ashabii/nepali_crop_data`** (Hugging Face). Run `data/extract_data.py` to download and extract it locally.

Full details on labelling, class balancing, splits, and augmentation are in the report's Data section (Section 4) and in `data/crop_disease_corpus_builder.ipynb`.

**Attribution:** if reusing this dataset, please cite the three original source datasets above in addition to this repository.

## Models

| Model | Type | Hugging Face Hub |
|---|---|---|
| Baseline CNN | From-scratch CNN classifier | `prathamshrestha69/CNN-baseline` |
| Proposed model | SmolVLM-256M-Instruct + LoRA | `w4ashabii/SmolVLM256M_CropDisease` |

## Demo Interface

A lightweight Gradio app (`demo/demo.ipynb`) lets a non-technical user upload a leaf photo and see both models' outputs at once, the CNN's crop prediction and SmolVLM's full crop-and-disease description in plain language. Built for a farmer or extension worker to actually try, not just read metrics about.

## How to Run

1. Clone this repo and install dependencies (PyTorch, `transformers`, `unsloth`, `huggingface_hub`).
2. Run `data/extract_data.py` to download the prepared dataset.
3. Run `notebooks/CNN-baseline.ipynb` to train/evaluate the baseline CNN, or download the pretrained checkpoint from `models/baseline_cnn_final/`.
4. Run `notebooks/smolvlm256m_nepali_crop_finetune.ipynb` to fine-tune SmolVLM-256M, or pull the published adapter from the Hugging Face Hub link above.
5. Run `results/crop_disease_model_evaluations.ipynb` to reproduce metrics, confusion matrices, and error analysis for both models.
6. Run `demo/demo.ipynb` to launch an interactive Gradio demo where you can upload a leaf photo and see both models' predictions side by side.

## Results

| Model | Task | Accuracy | Macro F1 |
|---|---|---|---|
| Baseline CNN | Crop classification (9 classes) | 91.67% | 91.66% |
| SmolVLM-256M + LoRA | Crop classification | 99.56% | 99.56% |
| SmolVLM-256M + LoRA | Disease classification (pooled across crops) | 98.33% | 90.32% |
| SmolVLM-256M + LoRA | Combined crop + disease (exact match) | 98.22% | n/a |

The CNN has no disease-detection capability. It only ever predicts a crop label, so disease evaluation is SmolVLM-only.

Note: test images for cherry, peach, pepper, and strawberry were topped up from the validation/train split due to insufficient held-out test images for those classes, so their reported accuracy should be treated as slightly optimistic (validation-augmented) compared to the other 5 crops, which had a full held-out test set.

Full metrics, confusion matrices (per-crop disease matrices included), and error analysis are in `results/crop_disease_model_evaluations.ipynb` and Section 6 of the full report.

## Ethics & Limitations

This system is intended to be **assistive only**, not a substitute for a qualified agricultural extension officer. Key limitations, including data provenance, imperfect label quality, language accessibility, and the digital divide, are discussed in full in Sections 7 and 8 of the report. No personally identifying data (people, farms, GPS coordinates) appears anywhere in the dataset.

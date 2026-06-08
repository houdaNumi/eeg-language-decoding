# EEG Language Decoding

EEG-based sentiment detection during natural reading using the ZuCo dataset.

## Objective
Predict the sentiment (positive / negative / neutral) of words from EEG brain 
signals recorded while subjects read natural sentences.

## Dataset
ZuCo — Zurich Cognitive Language Processing Corpus  
12 subjects, 400 sentences, 105 EEG electrodes, 8 frequency bands (theta, alpha, beta, gamma).

## Pipeline
1. **load_data.py** — Load and explore the .mat file structure
2. **features.py** — Extract TRT frequency band features per word → X (5007, 840), y (5007,)
3. **model.py** — Train SVM classifier with SelectKBest feature selection

## Results (1 subject, Task 1)
- Accuracy : **42.5%** (baseline random = 33%)
- Best class : neutral (F1 = 0.51)
- Weakest class : negative (F1 = 0.21)

## Stack
Python, scipy, numpy, matplotlib, scikit-learn

## Next Steps
- Add remaining 11 subjects
- Word-level sentiment labels
- Try KNN and other classifiers
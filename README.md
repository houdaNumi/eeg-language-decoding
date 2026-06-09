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
| Model | Accuracy | Negative F1 |
|---|---|---|
| SVM normal | 42.5% | 0.21 |
| SVM balanced | 41.8% | 0.35 ← best |
| KNN | 38.8% | 0.32 |

**Best model : SVM with class_weight='balanced'**

## Experiments
- Tested VADER for word-level labels → too imbalanced (88% neutral), abandoned
- Tested KNN → lower accuracy than SVM
- Applied class_weight='balanced' → better detection of negative class


## Stack
Python, scipy, numpy, matplotlib, scikit-learn 


## Next Steps
- Add remaining 11 subjects
- Improve feature selection (try different k values)



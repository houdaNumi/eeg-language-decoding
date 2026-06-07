# EEG Language Decoding

Detecting sentiment (positive / negative / neutral) from EEG brain signals recorded during natural reading.

## Dataset
ZuCo — Zurich Cognitive Language Processing Corpus  
12 subjects, 400 sentences, 105 EEG electrodes, 8 frequency bands.

## Progress
- Loaded and explored the .mat data structure
- Understood the nested structure (sentenceData → sentence → word → EEG features)
- Visualized raw EEG signal of a single electrode during word reading
- Accessed TRT frequency band features per word (theta, alpha, beta, gamma)

## Stack
Python, scipy, numpy, matplotlib
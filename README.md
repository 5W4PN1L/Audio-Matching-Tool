## Overview
This project focuses on enhancing audio similarity detection using machine learning and signal processing techniques. By leveraging Python libraries like **Librosa** and **Resemblyzer**, the system extracts voice embeddings and computes cosine similarity between audio files. The solution addresses the challenge of distinguishing between similar voices while accounting for noise variations. Advanced feature extraction techniques such as **MFCC** and **chroma features** are employed to improve accuracy. The model is fine-tuned to identify the same speaker's voice across noisy and clean audio samples.

## Features
- **Audio preprocessing** and feature extraction using **Librosa**
- **Voice embedding** generation with **Resemblyzer**
- **Cosine similarity** calculation for audio comparison
- **MFCC** and **chroma feature** extraction to enhance comparison accuracy
- Robust to **noisy datasets**
- Visualization of **waveforms**, **spectrograms**, and **similarity scores**

## Requirements
To install the required libraries, run:
```bash
pip install -r requirements.txt

## Libraries Used:
- **numpy**
- **librosa**
- **seaborn**
- **matplotlib**
- **resemblyzer**

## Usage
1. Place your target audio file and reference audio files in the project directory.
2. Run the `audio_similarity.py` script to compute the similarity between the target and reference audio files.

To execute the script:
```bash
python audio_similarity.py


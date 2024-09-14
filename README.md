
# Overview

This project focuses on enhancing audio similarity detection using machine learning and signal processing techniques. By leveraging Python libraries like Librosa and Resemblyzer, the system extracts voice embeddings and computes cosine similarity between audio files. The solution addresses the challenge of distinguishing between similar voices while accounting for noise variations. Advanced feature extraction techniques such as MFCC and chroma features are employed to improve accuracy. The model is fine-tuned to identify the same speaker's voice across noisy and clean audio samples.

# Features

- **Audio preprocessing** and feature extraction using **Librosa**
- **Voice embedding** generation with **Resemblyzer**
- **Cosine similarity** calculation for audio comparison
- **MFCC** and **chroma feature** extraction to enhance comparison accuracy
- Robust to **noisy datasets**
- Visualization of **waveforms**, **spectrograms**, and **similarity scores**



## Requirements

Python 3.7 or later is recommended.



## Installation

Install the required libraries by running

```bash
  pip install numpy

```
```bash
  pip install librosa

```
```bash
  pip install seaborn

```
```bash
  pip install matplotlib

```
```bash
  pip install resemblyzer

```

Download or clone the repository from GitHub.
```bash
  git clone https://github.com/5W4PN1L/Audio-Matching-Tool.git

```
## Usage

1. Save the audio files in `.wav` format.
2. Place your target audio file in the project directory and name it `aud1.wav`, or modify the script to use your own file name.
3. Place reference audio files (e.g., `audio1.wav`, `audio4.wav`) in the same directory, or modify the script to use your own file name.
4. Run the `audiomatching.py` script to compute the similarity between the target and reference audio files.


 
To execute the script:

```bash
  python audiomatching.py
```
## Authors

- [B.Swapnil](https://www.linkedin.com/in/b-swapnil-505a85251)

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import librosa
import librosa.display
from resemblyzer import VoiceEncoder, preprocess_wav
from pathlib import Path
from typing import List

def get_embedding(file_path: Path) -> np.ndarray:
    wav = preprocess_wav(file_path)
    encoder = VoiceEncoder()
    return encoder.embed_utterance(wav)

def calculate_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

def find_most_similar(target_audio_path: Path, reference_audio_paths: List[Path]) -> (int, List[float]):
    target_embedding = get_embedding(target_audio_path)
    reference_embeddings = [get_embedding(path) for path in reference_audio_paths]
    similarities = [calculate_similarity(target_embedding, ref_embed) for ref_embed in reference_embeddings]
    for i, score in enumerate(similarities):
        print(f"Similarity with {reference_audio_paths[i].name}: {score * 100:.2f}%")
    most_similar_index = np.argmax(similarities)
    return most_similar_index, similarities

def plot_all_data(target_audio_path: Path, reference_audio_paths: List[Path], similarities: List[float], most_similar_index: int):
    num_files = len(reference_audio_paths) + 1
    fig, axs = plt.subplots(num_files, 3, figsize=(20, 5 * num_files))
    target_wav, target_sr = librosa.load(target_audio_path, sr=None)
    librosa.display.waveshow(target_wav, sr=target_sr, ax=axs[0, 0])
    axs[0, 0].set_title(f'Waveform of {target_audio_path.name}')
    
    target_S = librosa.stft(target_wav)
    target_S_db = librosa.amplitude_to_db(np.abs(target_S), ref=np.max)
    librosa.display.specshow(target_S_db, sr=target_sr, x_axis='time', y_axis='log', ax=axs[0, 1])
    axs[0, 1].set_title(f'Spectrogram of {target_audio_path.name}')
    axs[0, 2].axis('off')
    axs[0, 2].text(0.5, 0.5, f'Most Similar: {reference_audio_paths[most_similar_index].name}\nSimilarity Score: {similarities[most_similar_index] * 100:.2f}%', 
                   ha='center', va='center', fontsize=14, bbox=dict(facecolor='white', alpha=0.5))
    for i, file_path in enumerate(reference_audio_paths):
        wav, sr = librosa.load(file_path, sr=None)
        librosa.display.waveshow(wav, sr=sr, ax=axs[i+1, 0])
        axs[i+1, 0].set_title(f'Waveform of {file_path.name}')
        S = librosa.stft(wav)
        S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
        librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log', ax=axs[i+1, 1])
        axs[i+1, 1].set_title(f'Spectrogram of {file_path.name}')
        similarity = similarities[i]
        axs[i+1, 2].barh([0], [similarity], align='center')
        axs[i+1, 2].set_xlim(0, 1)
        axs[i+1, 2].set_yticks([])
        axs[i+1, 2].set_title(f'Similarity: {similarity * 100:.2f}%')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    target_audio = "aud1.wav"
    reference_audios = ["audio2.wav", "audio4.wav"]
    target_audio_path = Path(target_audio)
    reference_audio_paths = [Path(file) for file in reference_audios]
    most_similar_index, similarity_scores = find_most_similar(target_audio_path, reference_audio_paths)
    print(f"\nThe target audio is most similar to reference audio file '{reference_audios[most_similar_index]}' with a similarity score of {similarity_scores[most_similar_index] * 100:.2f}%.")
    plot_all_data(target_audio_path, reference_audio_paths, similarity_scores, most_similar_index)

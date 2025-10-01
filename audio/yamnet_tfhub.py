#!/usr/bin/env python
import argparse
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import soundfile as sf
import pandas as pd
import urllib.request
import io


MODEL_URL = "https://tfhub.dev/google/yamnet/1"
CLASS_MAP_URL = "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv"


@tf.function
def _model_infer(m, audio):
    scores, embeddings, spectrogram = m(audio)
    return scores


def load_wav_mono(wav_path, target_sr=16000):
    wav, sr = sf.read(wav_path)
    if wav.ndim > 1:
        wav = np.mean(wav, axis=1)
    if sr != target_sr:
        import librosa
        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
    return wav.astype(np.float32), target_sr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", required=True)
    parser.add_argument("--topk", type=int, default=5)
    args = parser.parse_args()

    yamnet = hub.load(MODEL_URL)
    wav, sr = load_wav_mono(args.wav)
    audio_tensor = tf.convert_to_tensor(wav, dtype=tf.float32)
    scores = _model_infer(yamnet, audio_tensor)
    scores_np = scores.numpy()
    mean_scores = np.mean(scores_np, axis=0)

    # подгружаем список классов
    with urllib.request.urlopen(CLASS_MAP_URL) as resp:
        class_map_csv = resp.read()
    class_df = pd.read_csv(io.BytesIO(class_map_csv))
    classes = class_df["display_name"].tolist()

    top_idx = np.argsort(mean_scores)[-args.topk:][::-1]
    for i in top_idx:
        print(f"{classes[i]}: {mean_scores[i]:.4f}")


if __name__ == "__main__":
    main()

# python yamnet_tfhub.py --wav data/morse.wav

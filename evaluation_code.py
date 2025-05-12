import numpy as np
from python_speech_features import mfcc
from scipy.spatial.distance import euclidean
import librosa
from pesq import pesq
from pystoi import stoi
import os

def compute_mcd(ref_audio_path, synth_audio_path, sr=16000):
    ref, _ = librosa.load(ref_audio_path, sr=sr)
    synth, _ = librosa.load(synth_audio_path, sr=sr)

    # Trim both to the same length
    min_len = min(len(ref), len(synth))
    ref = ref[:min_len]
    synth = synth[:min_len]

    ref_mfcc = mfcc(ref, sr)
    synth_mfcc = mfcc(synth, sr)

    min_mfcc_len = min(len(ref_mfcc), len(synth_mfcc))
    distance = np.mean([
        euclidean(ref_mfcc[i], synth_mfcc[i]) for i in range(min_mfcc_len)
    ])
    return distance

def compute_pesq(ref_path, synth_path):
    ref, sr = librosa.load(ref_path, sr=16000)
    synth, _ = librosa.load(synth_path, sr=16000)

    # Trim to same length
    min_len = min(len(ref), len(synth))
    ref = ref[:min_len]
    synth = synth[:min_len]

    return pesq(sr, ref, synth, 'wb')  # 'wb' = wideband

def compute_stoi(ref_path, synth_path):
    ref, sr = librosa.load(ref_path, sr=16000)
    synth, _ = librosa.load(synth_path, sr=16000)

    # Trim to same length
    min_len = min(len(ref), len(synth))
    ref = ref[:min_len]
    synth = synth[:min_len]

    return stoi(ref, synth, sr, extended=False)

# Paths to evaluation data
refs = sorted(os.listdir("evaluation/ground_truth"))
synths = sorted(os.listdir("evaluation/synthesized"))

mcd_scores, pesq_scores, stoi_scores = [], [], []

# Compute metrics
for fname in refs:
    ref = os.path.join("evaluation/ground_truth", fname)
    syn = os.path.join("evaluation/synthesized", fname)

    mcd_scores.append(compute_mcd(ref, syn))
    pesq_scores.append(compute_pesq(ref, syn))
    stoi_scores.append(compute_stoi(ref, syn))

# Report averages
print("Avg MCD:", sum(mcd_scores)/len(mcd_scores))
print("Avg PESQ:", sum(pesq_scores)/len(pesq_scores))
print("Avg STOI:", sum(stoi_scores)/len(stoi_scores))

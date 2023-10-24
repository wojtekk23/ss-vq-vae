# Copyright 2020 InterDigital R&D and Télécom Paris.
# Author: Ondřej Cífka
# License: Apache 2.0

import bidict
import numpy as np
import torch
import librosa


def collate_padded_tuples(batch):
    batch = tuple(zip(*batch))
    lengths = [[x.shape[1] for x in inputs] for inputs in batch]
    max_lengths = [max(x) for x in lengths]
    batch = [np.array([np.pad(x, [(0, 0), (0, max(0, max_len - x.shape[1]))]) for x in inputs]) for inputs, max_len in zip(batch, max_lengths)]
    # batch = [[np.pad(x, [(0, 0), (0, max(0, max_len - x.shape[1]))]) for x in inputs]
    #          for inputs, max_len in zip(batch, max_lengths)]
    return tuple((torch.as_tensor(x), torch.as_tensor(l)) for x, l in zip(batch, lengths))


def markdown_format_code(text):
    return '    ' + text.replace('\n', '\n    ')

def generate_chromagram(audio, sample_rate=16_000, hop_length=512, n_fft=2048):
    chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate, hop_length=hop_length, n_fft=n_fft)
    return chroma

# Function to calculate Mean Squared Error (MSE) between two chromagrams
def chroma_mse(chroma1, chroma2):
    # Make sure both chromagrams have the same shape
    min_length = min(chroma1.shape[1], chroma2.shape[1])
    chroma1 = chroma1[:, :min_length]
    chroma2 = chroma2[:, :min_length]
    
    return np.mean((chroma1 - chroma2)**2)

def compare_chroma(output, reference, sr=16_000):
    chroma_output, chroma_reference = generate_chromagram(output, sample_rate=sr), generate_chromagram(reference, sample_rate=sr)
    return chroma_mse(chroma_output, chroma_reference)
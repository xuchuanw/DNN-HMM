import pandas as pd
from pandas import DataFrame, Series
import numpy as np
from python_speech_features import *
from scipy.io import wavfile
import os
import glob
import sys
import librosa


def gen_data(data_src, data_dir):
    data_dirname = os.path.dirname(data_dir)
    if not os.path.exists(data_dirname):
        os.mkdir(data_dirname)
    tmp = {}
    file_path = data_src + "/*/*.wav"
    utt2feats_scp = DataFrame([], columns=['utt', 'feats', 'label'])
    for utt in glob.glob(file_path):

        feats = extract_mfcc(utt)
        basename = os.path.dirname(utt)
        label = os.path.basename(basename)

        tmp['label'] = label

        tmp['utt'] = utt
        tmp['feats'] = feats
        tmp['length'] = feats.shape[0]

        utt2feats_scp = utt2feats_scp.append(tmp, ignore_index=True)
    utt2feats_scp.to_pickle(data_dir)


def compute_mfcc(file):

    fs, audio = wavfile.read(file)

    mfcc_feat = mfcc(audio, samplerate=fs, numcep=13, winlen=0.025, winstep=0.01, nfilt=26, nfft=2048, lowfreq=0,
                     highfreq=None, preemph=0.97)
    d_mfcc_feat = delta(mfcc_feat, 1)
    d_mfcc_feat2 = delta(mfcc_feat, 2)
    # mfcc_feat = mfcc_feat + d_mfcc_feat
    feature_mfcc = np.hstack((mfcc_feat, d_mfcc_feat, d_mfcc_feat2))

    # print(mfcc_feat)
    # feature_mfcc = np.hstack((mfcc_feat))
    # print(feature_mfcc)
    return feature_mfcc


def extract_mfcc(full_audio_path):
    wave, sample_rate = librosa.load(full_audio_path, mono=True)
    mfcc_features = librosa.feature.mfcc(wave, sample_rate)
    return mfcc_features.T



if __name__ == '__main__':
    gen_data('./english_digit/train', './feats/english_train_lib.pkl')
    gen_data('./english_digit/test', './feats/english_test_lib.pkl')

    print("Feature extraction done!")

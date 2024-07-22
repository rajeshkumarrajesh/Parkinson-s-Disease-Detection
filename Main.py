import pandas as pd
import biosppy.signals.ecg as ecg
import random as rn
from librosa.feature import spectral_centroid
from BFGO import BFGO
from DMO import DMO
from Model_CNN import Model_CNN
from Model_NN import Model_NN
from Model_PDCNNet import Model_PDCNNet
from Model_ResLSTM import Model_ResLSTM
from Plot_Results import *
from RHA import RHA
from THDN import THDN
from Global_Vars import Global_Vars
from Model_1DCNN_Feat import Model_1DCNN_Feat
from Model_Auto import Model_AutoEn_Feat
from Model_RNN_Feat import Model_RNN_Feat
from POA import POA
from Proposed import Proposed
from objfun_feat import objfun
from Spectral_Features import density, rms, zcr
from Spectral_Flux import spectralFlux
from scipy.signal import find_peaks
import librosa
from numpy import matlib


def extract_Wave_features(signal, sampling_rate):
    qrs_indices = ecg.hamilton_segmenter(signal=signal, sampling_rate=sampling_rate)[0]
    qrs_features = ecg.extract_heartbeats(signal=signal, rpeaks=qrs_indices, sampling_rate=sampling_rate)
    p_wave_features = qrs_features['templates'][:, :int(0.1 * sampling_rate)]
    t_wave_features = qrs_features['templates'][:, int(0.4 * sampling_rate):int(0.6 * sampling_rate)]
    u_wave_features = qrs_features['templates'][:, int(0.5 * sampling_rate):]
    waves = [p_wave_features[0], qrs_features[0][0], t_wave_features[0], u_wave_features[0]]
    shortest_list = min(waves, key=len)
    feature = [p_wave_features[0][:len(shortest_list)], qrs_features[0][0][:len(shortest_list)],
               t_wave_features[0][:len(shortest_list)], u_wave_features[0][:len(shortest_list)]]
    Feat = np.ravel(feature)
    return Feat


def Spectral_Feature(Audio):  # Single inputs
    Density = density(Audio)
    Flux = spectralFlux(Audio)
    zero_crossings = librosa.zero_crossings(Audio, pad=True)
    zero_crossing = sum(zero_crossings)
    peaks, _ = find_peaks(Audio, height=0)
    peak_amp = np.mean(peaks)
    Thdn = THDN(np.uint8(Audio), 44100)
    RMS = rms(Audio)
    ZCR = zcr(Audio)
    mfccs = librosa.feature.mfcc(y=Audio, sr=44100, n_mfcc=1)
    roll_off = librosa.feature.spectral_rolloff(y=Audio, sr=44100)
    spec = [Density, Flux, zero_crossing, peak_amp, Thdn, RMS, ZCR, np.mean(mfccs), np.mean(roll_off)]

    return spec


# Read Dataset
an = 0
if an == 1:
    df = pd.read_csv('./Dataset/Dataset_1/pd_EEG_features.csv')
    data = df.values[:, :-1]
    Target = df.values[:, -1]
    np.save('Data.npy', data)
    np.save('Target.npy', np.reshape(Target, (-1, 1)))

# Feature Extraction Using Wave Features
an = 0
if an == 1:
    Data = np.load('Data.npy', allow_pickle=True)
    Wave_Features = []
    fs = [300, 105]
    for i in range(len(Data)):
        print(i, len(Data))
        Wave_feat = extract_Wave_features(Data[i], fs[0])
        Wave_Features.append(Wave_feat)
    np.save('Wave_Feature.npy', np.asarray(Wave_Features))

# Feature Extraction Using Temporal Features
an = 0
if an == 1:
    Data = np.load('Data.npy', allow_pickle=True)
    Tar = np.load('Target.npy', allow_pickle=True)
    Feature = Model_1DCNN_Feat(Data, Tar)
    np.save('Temporal_Feat.npy', Feature)

# Feature Extraction Using Spatial Feature
an = 0
if an == 1:
    Data = np.load('Data.npy', allow_pickle=True)
    Tar = np.load('Target.npy', allow_pickle=True)
    Feature = Model_RNN_Feat(Data, Tar)
    np.save('Spatial_Feat.npy', Feature)

# Feature Extraction Using AutoEncoder
an = 0
if an == 1:
    Data = np.load('Data.npy', allow_pickle=True)
    Tar = np.load('Target.npy', allow_pickle=True)
    Feature = Model_AutoEn_Feat(Data, Tar)
    np.save('AutoEn_Feat.npy', Feature)

# Spectral Feature Extraction
an = 0
if an == 1:
    EEG_Signal = np.load('Data.npy', allow_pickle=True)
    EEG_feat = []
    for i in range(len(EEG_Signal)):
        print(i, len(EEG_Signal))
        EEG_Spect_features = Spectral_Feature(EEG_Signal[i])
        EEG_feat.append(EEG_Spect_features)
        np.save('Spectral_Feat.npy', EEG_feat)

# Optimization for Classification
an = 0
if an == 1:
    wave_Feat = np.load('Wave_Feature.npy', allow_pickle=True)
    temp_Feat = np.load('Temporal_Feat.npy', allow_pickle=True)
    spact_Feat = np.load('Spectral_Feat.npy', allow_pickle=True)
    spat_Feat = np.load('Spatial_Feat.npy', allow_pickle=True)
    auto_Feat = np.load('AutoEn_Feat.npy', allow_pickle=True)
    Target = np.load('Target.npy', allow_pickle=True)
    Global_Vars.wave_Feat = wave_Feat
    Global_Vars.temp_Feat = temp_Feat
    Global_Vars.spact_Feat = spact_Feat
    Global_Vars.spat_Feat = spat_Feat
    Global_Vars.auto_Feat = auto_Feat
    Global_Vars.Target = Target
    Npop = 10
    Chlen = 5
    xmin = matlib.repmat((0.01 * np.ones((1, Chlen))), Npop, 1)
    xmax = matlib.repmat((0.99 * np.ones((1, Chlen))), Npop, 1)
    initsol = np.zeros(xmin.shape)
    for i in range(xmin.shape[0]):
        for j in range(xmin.shape[1]):
            initsol[i, j] = rn.uniform(xmin[i, j], xmax[i, j])
    fname = objfun
    max_iter = 50

    print("DMO...")
    [bestfit1, fitness1, bestsol1, time] = DMO(initsol, fname, xmin, xmax, max_iter)  # DMO

    print("BFGO...")
    [bestfit2, fitness2, bestsol2, time1] = BFGO(initsol, fname, xmin, xmax, max_iter)  # BFGO

    print("RHA...")
    [bestfit3, fitness3, bestsol3, time2] = RHA(initsol, fname, xmin, xmax, max_iter)  # RHA

    print("POA...")
    [bestfit4, fitness4, bestsol4, time3] = POA(initsol, fname, xmin, xmax, max_iter)  # POA

    print("Proposed...")
    [bestfit5, fitness5, bestsol5, time4] = Proposed(initsol, fname, xmin, xmax, max_iter)  # Proposed

    BestSol = [bestsol1, bestsol2, bestsol3, bestsol4, bestsol5]
    np.save('BestSol_CLS.npy', BestSol)

# Classification
an = 0
if an == 1:
    Feat_1 = np.load('Wave_Feature.npy', allow_pickle=True)  # loading step
    Feat_2 = np.load('Temporal_Feat.npy', allow_pickle=True)
    Feat_3 = np.load('AutoEn_Feat.npy', allow_pickle=True)
    Feat_4 = np.load('Spectral_Feat.npy', allow_pickle=True)
    Feat_5 = np.load('Spatial_Feat.npy', allow_pickle=True)
    Target = np.load('Target.npy', allow_pickle=True)  # loading step
    BestSol = np.load('BestSol_CLS.npy', allow_pickle=True)
    Feature = Feat_1
    K = 5
    Per = 1 / 5
    Perc = round(Feature.shape[1] * Per)
    eval = []
    for i in range(K):
        Eval = np.zeros((10, 14))
        for j in range(5):
            Feat = Feature
            sol = np.round(BestSol[j, :]).astype(np.int16)
            Test_Data = Feat[i * Perc: ((i + 1) * Perc), :]
            Test_Target = Target[i * Perc: ((i + 1) * Perc), :]
            test_index = np.arange(i * Perc, ((i + 1) * Perc))
            total_index = np.arange(Feat.shape[0])
            train_index = np.setdiff1d(total_index, test_index)
            Train_Data = Feat[train_index, :]
            Train_Target = Target[train_index, :]
            Eval[j, :] = Model_ResLSTM(Feat_1, Feat_2, Feat_3, Feat_4, Feat_5, Target, sol)  # RconvLSTM With optimization
        Eval[5, :], pred = Model_PDCNNet(Train_Data, Train_Target, Test_Data, Test_Target)  # Model PDCNNet
        Eval[7, :], pred1 = Model_NN(Train_Data, Train_Target, Test_Data, Test_Target)  # Model NN
        Eval[8, :], pred2 = Model_CNN(Train_Data, Train_Target, Test_Data, Test_Target)  # Model CNN
        Eval[9, :], pred3 = Model_ResLSTM(Feat_1, Feat_2, Feat_3, Feat_4, Feat_5, Target)  # RconvLSTM  Without optimization
        Eval[9, :] = Eval[4, :]
        eval.append(Eval)
    np.save('Eval_KFold.npy', eval)  # Save Eval

plotConvResults()
plot_results()
Plot_ROC_Curve()

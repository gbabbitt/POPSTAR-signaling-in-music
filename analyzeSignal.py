#!/usr/bin/env python

#############################################################################
######   POPSTAR software for detecting fitness signaling in music
######   produced by Dr. Gregory A. Babbitt
######   and students at the Rochester Instituteof Technology in 2025.
######   Offered freely without guarantee.  License under GPL v3.0
#############################################################################


import getopt, sys # Allows for command line arguments
import os
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
#from plotnine.data import mpg
from scipy.io import wavfile
from scipy import signal
from scipy.signal import find_peaks
from statsmodels.tsa.stattools import acf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.stattools import adfuller
### peak parameters ###
HT = 0.05  # heigth
WD = None  # width
DIST = 50  # distance
#######################
import math
import random
bootstp = 50
import random as rnd
#import pytraj as pt
#import nglview as nv
from scipy.spatial import distance
from scipy.stats import entropy
from scipy.stats import ks_2samp
from scipy.stats import f_oneway
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.decomposition import TruncatedSVD
from sklearn import metrics
from hurst import compute_Hc, random_walk
import re
# for ggplot
from plotnine import *
from pydub import AudioSegment
import soundfile
import librosa 
import multiprocessing

# IMPORTANT NOTE - run in base conda env, not in atomdance conda env   
################################################################################
# find number of cores
num_cores = multiprocessing.cpu_count()

# read popstar ctl file
infile = open("popstar.ctl", "r")
infile_lines = infile.readlines()
for x in range(len(infile_lines)):
    infile_line = infile_lines[x]
    #print(infile_line)
    infile_line_array = str.split(infile_line, ",")
    header = infile_line_array[0]
    value = infile_line_array[1]
    #print(header)
    #print(value)
    if(header == "name"):
        name = value
        print("my file/folder name is",name)
    if(header == "int"):
        tm = value
        print("my time interval is",tm)
    if(header == "input"):
        fof = value
        print("file or folder is",fof)
    if(header == "lyrics"):
        lyr = value
        print("lyrics present is",lyr)
    if(header == "max"):
        maxOpt = value
        print("use max values",maxOpt)
    if(header == "metro"):
        met = value
        print("my metronome option is",met)    
infile.close()
 ###### variable assignments ######
inp = ""+name+""
tm = int(tm)
fof = ""+fof+""
lyr = ""+lyr+""
maxOpt = ""+maxOpt+""
fileORfolder = fof
met = ""+met+""

if(fof=="file"):
    # calculate number of faces for single file
    lst = os.listdir("%s_analysis/intervals/" % inp) # your directory path
    face_num = int(len(lst)/4)  # note folder has 4 types of files
    print("number of Chernoff faces is %s" % face_num)
#####################################################################
def create_file_lists():   
    folder_path = "%s_analysis/intervals/" % inp
    print(folder_path)
    global sound_file_paths
    sound_file_paths = []  # for .wav files
    global data_file_paths
    data_file_paths = [] # for .dat files
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, "%s" % (filename))
        #print(file_path)
        if os.path.isfile(file_path) and file_path[-4:] == ".wav":
            print("generating sound file list for %s" % (filename))
            sound_file_paths.append(file_path)
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, "%s" % (filename))
        #print(file_path)
        if os.path.isfile(file_path) and file_path[-4:] == ".dat":
            print("generating data file list for %s" % (filename))
            data_file_paths.append(file_path)
    #print(sound_file_paths)
    #print(data_file_paths)
    return sound_file_paths
    return data_file_paths

def create_file_lists_batch(): 
    folder_path1 = "%s_analysis/intervals/" % inp
    #print(folder_path1)
    global sound_file_paths
    sound_file_paths = []  # for .wav files
    global data_file_paths
    data_file_paths = [] # for .dat files
    for foldername in os.listdir(folder_path1):
        folder_path2 = os.path.join(folder_path1, "%s" % (foldername))
        print(folder_path2)
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, "%s" % (filename))
            #print(file_path)
            if os.path.isfile(file_path) and file_path[-4:] == ".wav":
                print("generating sound file list for %s" % (filename))
                sound_file_paths.append(file_path)
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, "%s" % (filename))
            #print(file_path)
            if os.path.isfile(file_path) and file_path[-4:] == ".dat":
                print("generating data file list for %s" % (filename))
                data_file_paths.append(file_path)
        #print(sound_file_paths)
        #print(data_file_paths)
        return sound_file_paths
        return data_file_paths
        
########################################################################

# surprise metrics
def nvi_stat(item):
    path_objs = item.split("/")
    filename = path_objs[2]
    writePath = "%s_analysis/NVIvalues.txt" % (inp)
    df_in = pd.read_csv(item, delimiter='\t',header=None)
    txt_out = open(writePath, 'a')
    shp = df_in.shape
    #print(shp)
    n_cols = shp[1]
    n_rows = shp[0]
    #print(n_cols)
    NVIsum = 0
    for i in range(n_cols-1):
        #print("data in column %s" % i)
        #print(df_in[i])
        for j in range(n_cols-1):
            #print("data in column %s" % j)
            #print(df_in[j])
            # correlate
            #print("correlating columns %s %s" % (i,j))
            myCorr = np.corrcoef(df_in[i],df_in[j])
            myCorr = abs(myCorr[0,1])  # set to [0,0] and NVI should = 0
            #print(myCorr)
            NVIsum = NVIsum + (1-myCorr)
            #print(NVIsum)
    # normalize NVI to number of notes (i.e. columns)
    NVI = NVIsum/(n_cols*(n_cols-1))
    print("NVI (note variability index) = %s for %s" % (NVI,filename))
    txt_out.write("%s,%s\n" % (filename,NVI))
    txt_out.close

def lzc_stat(item):
    path_objs = item.split("/")
    filename = path_objs[2]
    writePath = "%s_analysis/LZCvalues.txt" % (inp)
    txt_out = open(writePath, 'a')
    infile = item
    with open(infile, 'rb') as audio_file: 
        binary_data = audio_file.read()
    hex_data = binary_data.hex()
    #print(hex_data)
    sequence = hex_data
    sub_strings = set()
    index = 0
    while index < len(sequence):
        for length in range(1, len(sequence) - index + 1):
            sub = sequence[index:index + length]
            if sub not in sub_strings:
                sub_strings.add(sub)
                index += length
                break
        else:
            break
    LZC = np.log(len(sub_strings))
    print("LZC (Lempel-Ziv complexity) = %s for %s" % (LZC,filename))
    txt_out.write("%s,%s\n" % (filename,LZC))
    txt_out.close
        

def adf_stat(item):
    path_objs = item.split("/")
    filename = path_objs[2]
    writePath = "%s_analysis/ADFvalues.txt" % (inp)
    txt_out = open(writePath, 'a')
    infile = item
    samplingFrequency, signalData = wavfile.read(infile)
    #print(signalData[:,1])
    #print(samplingFrequency)
    # Matplotlib.pyplot.specgram() function to
    # generate spectrogram
    if(signalData.ndim != 1):
        #print("flattening signal to 1D")
        signalData = signalData[:,1]
    signalData = np.float32(signalData)
    ADFdata_len = len(signalData)
    #print(ADFdata_len)
    if(ADFdata_len <= 500000):
        print("analyzing full signal")
        ADFdata = signalData
        ADFtest = adfuller(ADFdata, autolag='BIC')
    else:
        print("...analyzing only first 500000 elements of the signal")
        ADFdata = signalData[:500000]
        ADFtest = adfuller(ADFdata, autolag='BIC')
    #print(ADFtest)
    ADFtestStat = ADFtest[0]
    ADFpValue = ADFtest[1]
    print("ADF (augmented Dickey Fuller test) = %s for %s" % (ADFtestStat,filename))
    txt_out.write("%s,%s\n" % (filename,ADFtestStat))
    txt_out.close
    
def mli_stat(item):
    path_objs = item.split("/")
    filename = path_objs[2]
    writePath = "%s_analysis/MLIvalues.txt" % (inp)
    txt_out = open(writePath, 'a')
    infile = item
    samplingFrequency, signalData = wavfile.read(infile)
    #print(signalData[:,1])
    #print(samplingFrequency)
    # Matplotlib.pyplot.specgram() function to
    # generate spectrogram
    if(signalData.ndim != 1):
        #print("flattening signal to 1D")
        signalData = signalData[:,1]
    signalData = np.float32(signalData)
    # Hurst Exponent (measure memory 0 = negative memry, 0.5 = no memory, 1 = positive memory)
    H, c, data = compute_Hc(signalData, kind='change', simplified=True)
    #print("Hurst Exp = %s" % str(H))
    mem_level = 1-(2*abs(H-0.5)) # rescale 0-1  
    print("MLI (inverse memory level index) = %s for %s" % (mem_level,filename))
    txt_out.write("%s,%s\n" % (filename,mem_level))
    txt_out.close
    
# control metrics
def f0_var_stat(item):
    path_objs = item.split("/")
    filename = path_objs[2]
    writePath = "%s_analysis/FFVvalues.txt" % (inp)
    txt_out = open(writePath, 'a')
    infile = item
    y, sr = librosa.load(infile)
    f0, voiced_flag, voiced_probs = librosa.pyin(y,sr=sr,fmin=librosa.note_to_hz('C2'),fmax=librosa.note_to_hz('C7'))
    times = librosa.times_like(f0, sr=sr)
    #pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    #print(f0)
    #print(times)
    notes = []
    sum_diff = 0
    ##################
    for freq in f0:
        #print(freq)
        if(freq == "nan"):
            continue
        # equal tempered scale - middle octave range (octave 4)
        if(freq < 269.405 and freq >= 253.855): 
            note = "C"
            notes.append(note)
            sum_diff = sum_diff + abs(freq - 261.63)
        if(freq < 285.42 and freq >= 269.405):
            note = "C#"
            notes.append(note)
            sum_diff = sum_diff + abs(freq - 277.18)
        if(freq < 302.395 and freq >= 285.42):
            note = "D"
            notes.append(note)
            sum_diff = sum_diff + abs(freq - 293.66)
        if(freq < 320.38 and freq >= 302.395):
            note = "D#"
            notes.append(note)
            sum_diff = sum_diff + abs(freq - 311.13)    
        if(freq < 339.43 and freq >= 320.38):
            note = "E"
            notes.append(note)
            sum_diff = sum_diff + abs(freq - 329.63)    
        if(freq < 359.61 and freq >= 339.43):
            note = "F"
            notes.append(note)
            sum_diff = sum_diff + abs(freq - 349.23)    
        if(freq < 380.995 and freq >= 359.61):
            note = "F#"
            notes.append(note)
            sum_diff = sum_diff + abs(freq - 369.99)
        if(freq < 403.65 and freq >= 380.995):
            note = "G"
            notes.append(note)
            sum_diff = sum_diff + abs(freq - 392)
        if(freq < 427.65 and freq >= 403.65):
            note = "G#"
            notes.append(note)
            sum_diff = sum_diff + abs(freq - 415.3)
        if(freq < 453.08 and freq >= 427.65):
            note = "A"
            notes.append(note)
            sum_diff = sum_diff + abs(freq - 440)
        if(freq < 480.02 and freq >= 453.08):
            note = "A#"
            notes.append(note)
            sum_diff = sum_diff + abs(freq - 466.16) 
        if(freq < 507.74 and freq >= 480.02):
            note = "B"
            notes.append(note)
            sum_diff = sum_diff + abs(freq - 493.88) 
    ##############
    #print("notes detected")
    #print(notes)
    n_notes = len(notes)
    if(sum_diff == 0):
        FFV = 0
    if(sum_diff != 0):
        FFV = 1/((sum_diff/(len(notes)))+0.000001)
    print("FFV (f0 frequency control) = %s over %s notes for %s" % (FFV,n_notes,filename))
    txt_out.write("%s,%s,%s\n" % (filename,FFV,n_notes))
    txt_out.close
    
    
def fn_levels_stat(item):
    path_objs = item.split("/")
    filename = path_objs[2]
    writePath = "%s_analysis/HENvalues.txt" % (inp)
    txt_out = open(writePath, 'a')
    infile = item
    y, sr = librosa.load(infile)
    f0, voiced_flag, voiced_probs = librosa.pyin(y,sr=sr,fmin=librosa.note_to_hz('C2'),fmax=librosa.note_to_hz('C7'))
    S = np.abs(librosa.stft(y))
    times = librosa.times_like(S, sr=sr)
    # use the first 30 harmonics: 1, 2, 3, ..., 30
    harmonics = np.arange(1, 31)
    # And standard Fourier transform frequencies
    frequencies = librosa.fft_frequencies(sr=sr)
    harmonic_energy = librosa.f0_harmonics(S, f0=f0, harmonics=harmonics, freqs=frequencies) # 2D matrix
    #print(harmonic_energy)
    HEN = np.log(np.sum(harmonic_energy))
    print("HEN (harmonic energy) = %s over for %s" % (HEN,filename))
    txt_out.write("%s,%s\n" % (filename,HEN))
    txt_out.close

def beat_var(item):
    path_objs = item.split("/")
    filename = path_objs[2]
    writePath = "%s_analysis/BIVvalues.txt" % (inp)
    txt_out = open(writePath, 'a')
    writePath2 = "%s_analysis/EVIvalues.txt" % (inp)
    txt_out2 = open(writePath2, 'a')
    # Load an audio file
    infile = item
    y, sr = librosa.load(infile)
    # Estimate tempo and beat frames
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    # Convert beat frames to timestamps
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=1024)
    # Analyze beat spacing
    beat_intervals = np.diff(beat_times)
    mean_beat_interval = np.mean(beat_intervals)
    EVI = np.log(1/(((beat_intervals - mean_beat_interval)**2)+0.000001))
    EVI = np.sum(EVI)
    BIV = np.log(1/(np.var(beat_intervals)+0.000001))
    # Print the estimated tempo and beat intervals
    #print(f"Estimated tempo: {tempo} BPM")
    #print(f"Beat intervals: {beat_intervals}")
    print("BIV (beat interval control) = %s for %s" % (BIV,filename))
    print("EVI (beat interval evenness) = %s for %s" % (EVI,filename))
    txt_out.write("%s,%s\n" % (filename,BIV))
    txt_out.close
    txt_out2.write("%s,%s\n" % (filename,EVI))
    txt_out2.close

# energy metrics
def ampvar_stat(item):
    path_objs = item.split("/")
    filename = path_objs[2]
    writePath = "%s_analysis/AMPvalues.txt" % (inp)
    txt_out = open(writePath, 'a')
    infile = item
    samplingFrequency, signalData = wavfile.read(infile)
    #print(signalData[:,1])
    #print(samplingFrequency)
    # Matplotlib.pyplot.specgram() function to
    # generate spectrogram
    if(signalData.ndim != 1):
        #print("flattening signal to 1D")
        signalData = signalData[:,1]
    signalData = np.float32(signalData)
    #minVAL = np.min(signalData)
    #maxVAL = np.max(signalData)
    #norm_signalData = (signalData - minVAL) / (maxVAL-minVAL)
    AMP = np.log(np.var(signalData))
    print("AMP (amplitude variance) = %s for %s" % (AMP,filename))
    txt_out.write("%s,%s\n" % (filename,AMP))
    txt_out.close
    
def dimension_stat(item):
    path_objs = item.split("/")
    filename = path_objs[2]
    writePath = "%s_analysis/AC1values.txt" % (inp)
    txt_out = open(writePath, 'a')
    infile = item
    samplingFrequency, signalData = wavfile.read(infile)
    #print(signalData[:,1])
    #print(samplingFrequency)
    # Matplotlib.pyplot.specgram() function to
    # generate spectrogram
    if(signalData.ndim != 1):
        #print("flattening signal to 1D")
        signalData = signalData[:,1]
    signalData = np.float32(signalData)
    corr = signal.correlate(signalData, signalData)
    lags = signal.correlation_lags(len(signalData), len(signalData))
    corr = corr / np.max(corr) # normalize
   
    # remove self correlation = 1.0 at position 0
    mid_index = len(corr) // 2  # Floor division to get integer index
    if len(corr) % 2 == 0:  # Even number of elements
        middle = (mid_index - 1 + mid_index) / 2
    else:  # Odd number of elements
        middle = mid_index
    #print(middle)
    corr = np.delete(corr, middle)   
    lags = np.delete(lags, middle)
    
    # find max auto correlation
    lag = lags[np.argmax(corr)]
    #print(corr)
    #print(lags)
    MAC = np.max(corr)
    print("AC1 (1st order autocorrelation) = %s for %s" % (MAC,filename))
    txt_out.write("%s,%s\n" % (filename,MAC))
    txt_out.close
    
def tempo_stat(item):
    path_objs = item.split("/")
    filename = path_objs[2]
    writePath = "%s_analysis/TEMPOvalues.txt" % (inp)
    txt_out = open(writePath, 'a')
    infile = item
    song = AudioSegment.from_file(infile, format="wav") 
    #print(song.duration_seconds)
    dur = song.duration_seconds
    # Estimate the tempo (BPM)
    y, sr = librosa.load(item)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo = tempo[0]
    print("TEMPO (tempo - bpm) = %s for %s" % (tempo,filename))
    txt_out.write("%s,%s\n" % (filename,tempo))
    txt_out.close
    
def coll_data():
    print("collecting data")
    writePath = "%s_analysis/features_raw.txt" % (inp)
    txt_out = open(writePath, 'w')
    txt_out.write("file,AC1values,AMPvalues,BIVvalues,EVIvalues,FFVvalues,HENvalues,LZCvalues,MLIvalues,NVIvalues,TEMPOvalues\n")
    readPath1 = "%s_analysis/AC1values.txt" % (inp)
    txt_in1 = open(readPath1, 'r')
    readPath2 = "%s_analysis/AMPvalues.txt" % (inp)
    txt_in2 = open(readPath2, 'r')
    readPath3 = "%s_analysis/BIVvalues.txt" % (inp)
    txt_in3 = open(readPath3, 'r')
    readPath4 = "%s_analysis/EVIvalues.txt" % (inp)
    txt_in4 = open(readPath4, 'r')
    readPath5 = "%s_analysis/FFVvalues.txt" % (inp)
    txt_in5 = open(readPath5, 'r')
    readPath6 = "%s_analysis/HENvalues.txt" % (inp)
    txt_in6 = open(readPath6, 'r')
    readPath7 = "%s_analysis/LZCvalues.txt" % (inp)
    txt_in7 = open(readPath7, 'r')
    readPath8 = "%s_analysis/MLIvalues.txt" % (inp)
    txt_in8 = open(readPath8, 'r')
    readPath9 = "%s_analysis/NVIvalues.txt" % (inp)
    txt_in9 = open(readPath9, 'r')
    readPath10 = "%s_analysis/TEMPOvalues.txt" % (inp)
    txt_in10 = open(readPath10, 'r')
    AC1_lines = txt_in1.readlines()
    AMP_lines = txt_in2.readlines()
    BIV_lines = txt_in3.readlines()
    EVI_lines = txt_in4.readlines()
    FFV_lines = txt_in5.readlines()
    HEN_lines = txt_in6.readlines()
    LZC_lines = txt_in7.readlines()
    MLI_lines = txt_in8.readlines()
    NVI_lines = txt_in9.readlines()
    TEMPO_lines = txt_in10.readlines()
    length = len(AC1_lines)
    for i in range(len(AC1_lines)):
        for line in AC1_lines:
            line_split1 = line.split("_")
            seg_num = int(line_split1[0])
            line_split2 = line.split(",")
            if(i==seg_num):
                print("AC1 matching %s to %s" % (i,seg_num))
                file_name = line_split2[0]
                AC1 = float(line_split2[1])
        for line in AMP_lines:
            line_split1 = line.split("_")
            seg_num = int(line_split1[0])
            line_split2 = line.split(",")
            if(i==seg_num):
                print("AMP matching %s to %s" % (i,seg_num))
                file_name = line_split2[0]
                AMP= float(line_split2[1]) 
        for line in BIV_lines:
            line_split1 = line.split("_")
            seg_num = int(line_split1[0])
            line_split2 = line.split(",")
            if(i==seg_num):
                print("BIV matching %s to %s" % (i,seg_num))
                file_name = line_split2[0]
                BIV= float(line_split2[1]) 
        for line in EVI_lines:
            line_split1 = line.split("_")
            seg_num = int(line_split1[0])
            line_split2 = line.split(",")
            if(i==seg_num):
                print("EVI matching %s to %s" % (i,seg_num))
                file_name = line_split2[0]
                EVI= float(line_split2[1])
        for line in FFV_lines:
            line_split1 = line.split("_")
            seg_num = int(line_split1[0])
            line_split2 = line.split(",")
            if(i==seg_num):
                print("FFV matching %s to %s" % (i,seg_num))
                file_name = line_split2[0]
                FFV= float(line_split2[1])
        for line in HEN_lines:
            line_split1 = line.split("_")
            seg_num = int(line_split1[0])
            line_split2 = line.split(",")
            if(i==seg_num):
                print("HEN matching %s to %s" % (i,seg_num))
                file_name = line_split2[0]
                HEN= float(line_split2[1])
        for line in LZC_lines:
            line_split1 = line.split("_")
            seg_num = int(line_split1[0])
            line_split2 = line.split(",")
            if(i==seg_num):
                print("LZC matching %s to %s" % (i,seg_num))
                file_name = line_split2[0]
                LZC= float(line_split2[1])
        for line in MLI_lines:
            line_split1 = line.split("_")
            seg_num = int(line_split1[0])
            line_split2 = line.split(",")
            if(i==seg_num):
                print("MLI matching %s to %s" % (i,seg_num))
                file_name = line_split2[0]
                MLI= float(line_split2[1])
        for line in NVI_lines:
            line_split1 = line.split("_")
            seg_num = int(line_split1[0])
            line_split2 = line.split(",")
            if(i==seg_num):
                print("NVI matching %s to %s" % (i,seg_num))
                file_name = line_split2[0]
                NVI= float(line_split2[1])
        for line in TEMPO_lines:
            line_split1 = line.split("_")
            seg_num = int(line_split1[0])
            line_split2 = line.split(",")
            if(i==seg_num):
                print("TEMPO matching %s to %s" % (i,seg_num))
                file_name = line_split2[0]
                TEMPO= float(line_split2[1])
                
        txt_out.write("%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" % (file_name,AC1,AMP,BIV,EVI,FFV,HEN,LZC,MLI,NVI,TEMPO))
    txt_out.close


def norm_data():
    print("normalizing data")
    readPath = "%s_analysis/features_raw.txt" % (inp)
    writePath = "%s_analysis/features_norm.txt" % (inp)
    writePath2 = "%s_analysis/ternary.txt" % (inp)
    writePath3 = "%s_analysis/ternary_norm.txt" % (inp)
    df = pd.read_csv(readPath, delimiter=',',header=0)
    print(df)
    df = df.iloc[:, 1:]
    df_norm = df.apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    print(df_norm)
    with open(writePath, 'w') as txt_out:
        txt_out.write("AC1values,AMPvalues,BIVvalues,EVIvalues,FFVvalues,HENvalues,LZCvalues,MLIvalues,NVIvalues,TEMPOvalues\n")
        for index, row in df_norm.iterrows():
            line = ','.join(str("{:.8f}".format(x)) for x in row.values)  # Convert row to comma-separated string
            txt_out.write(line + '\n')  # Write line to file with newline character
        txt_out.close
    if(maxOpt == "no" and met == "yes"):
        df_energy = df_norm[['AC1values', 'AMPvalues', 'TEMPOvalues', 'HENvalues']].mean(axis=1)
        df_control = df_norm[['EVIvalues', 'FFVvalues', 'BIVvalues', 'HENvalues']].mean(axis=1)
        df_surprise = df_norm[['LZCvalues', 'MLIvalues', 'NVIvalues']].mean(axis=1)
    if(maxOpt == "yes" and met == "yes"):
        df_energy = df_norm[['AC1values', 'AMPvalues', 'TEMPOvalues', 'HENvalues']].max(axis=1)
        df_control = df_norm[['EVIvalues', 'FFVvalues', 'BIVvalues', 'HENvalues']].max(axis=1)
        df_surprise = df_norm[['LZCvalues', 'MLIvalues', 'NVIvalues']].max(axis=1)
    if(maxOpt == "no" and met == "no"):
        df_energy = df_norm[['AC1values', 'AMPvalues', 'TEMPOvalues', 'HENvalues']].mean(axis=1)
        df_control = df_norm[['FFVvalues', 'HENvalues']].mean(axis=1)
        df_surprise = df_norm[['LZCvalues', 'MLIvalues', 'NVIvalues']].mean(axis=1)
    if(maxOpt == "yes" and met == "no"):
        df_energy = df_norm[['AC1values', 'AMPvalues', 'TEMPOvalues', 'HENvalues']].max(axis=1)
        df_control = df_norm[['FFVvalues', 'HENvalues']].max(axis=1)
        df_surprise = df_norm[['LZCvalues', 'MLIvalues', 'NVIvalues']].max(axis=1)
    df_ternary = pd.concat([df_energy, df_control, df_surprise], axis=1)
    print(df_ternary)
    with open(writePath2, 'w') as txt_out:
        txt_out.write("energy,control,surprise\n")
        for index, row in df_ternary.iterrows():
            line = ','.join(str("{:.8f}".format(x)) for x in row.values)  # Convert row to comma-separated string
            txt_out.write(line + '\n')  # Write line to file with newline character
        txt_out.close
    df_ternary_norm = df_ternary.div(df_ternary.sum(axis=1), axis=0)
    print(df_ternary_norm)
    with open(writePath3, 'w') as txt_out:
        txt_out.write("energy,control,surprise\n")
        for index, row in df_ternary_norm.iterrows():
            line = ','.join(str("{:.8f}".format(x)) for x in row.values)  # Convert row to comma-separated string
            txt_out.write(line + '\n')  # Write line to file with newline character
        txt_out.close
    
    
#################################################################################
####################  main program      #########################################
#################################################################################
def main():
    if(fileORfolder == "file"):
        create_file_lists()
        print(data_file_paths)
        print(sound_file_paths)
    if(fileORfolder == "folder"):
        create_file_lists_batch()
        print(data_file_paths)
        print(sound_file_paths)
    
    ####################
    # energy metrics
    ####################
    print("calculating amplitude variance")
    writePath = "%s_analysis/AMPvalues.txt" % (inp)
    txt_out = open(writePath, 'w')
    txt_out.close
    with multiprocessing.Pool(processes=num_cores) as pool: # Use os.cpu_count() for max processes
        pool.map(ampvar_stat, sound_file_paths)
    print("calculating size/dimension")
    writePath = "%s_analysis/AC1values.txt" % (inp)
    txt_out = open(writePath, 'w')
    txt_out.close
    with multiprocessing.Pool(processes=num_cores) as pool: # Use os.cpu_count() for max processes
        pool.map(dimension_stat, sound_file_paths)
    print("calculating local tempo")
    writePath = "%s_analysis/TEMPOvalues.txt" % (inp)
    txt_out = open(writePath, 'w')
    txt_out.close
    with multiprocessing.Pool(processes=num_cores) as pool: # Use os.cpu_count() for max processes
        pool.map(tempo_stat, sound_file_paths)
    
    ####################
    # control metrics
    ####################
    print("calculating f0 variance")
    writePath = "%s_analysis/FFVvalues.txt" % (inp)
    txt_out = open(writePath, 'w')
    txt_out.close
    with multiprocessing.Pool(processes=num_cores) as pool: # Use os.cpu_count() for max processes
        pool.map(f0_var_stat, sound_file_paths)
    print("calculating fn harmonic energy")
    writePath = "%s_analysis/HENvalues.txt" % (inp)
    txt_out = open(writePath, 'w')
    txt_out.close
    with multiprocessing.Pool(processes=num_cores) as pool: # Use os.cpu_count() for max processes
        pool.map(fn_levels_stat, sound_file_paths)
    print("calculating beat interval variance")
    writePath = "%s_analysis/BIVvalues.txt" % (inp)
    txt_out = open(writePath, 'w')
    txt_out.close
    writePath2 = "%s_analysis/EVIvalues.txt" % (inp)
    txt_out2 = open(writePath2, 'w')
    txt_out2.close
    with multiprocessing.Pool(processes=num_cores) as pool: # Use os.cpu_count() for max processes
        pool.map(beat_var, sound_file_paths)
            
    ####################
    # surprise metrics
    ####################
    print("calculating Lempel-Ziv complexity")
    writePath = "%s_analysis/LZCvalues.txt" % (inp)
    txt_out = open(writePath, 'w')
    txt_out.close
    with multiprocessing.Pool(processes=num_cores) as pool: # Use os.cpu_count() for max processes
        pool.map(lzc_stat, sound_file_paths)
    print("calculating MLI statistic")
    writePath = "%s_analysis/MLIvalues.txt" % (inp)
    txt_out = open(writePath, 'w')
    txt_out.close
    with multiprocessing.Pool(processes=num_cores) as pool: # Use os.cpu_count() for max processes
        pool.map(mli_stat, sound_file_paths)
    print("calculating NVI statistic (zero order)")
    print("(Sawant et al. 2021 in MEE-BES)")
    writePath = "%s_analysis/NVIvalues.txt" % (inp)
    txt_out = open(writePath, 'w')
    txt_out.close
    with multiprocessing.Pool(processes=num_cores) as pool: # Use os.cpu_count() for max processes
        pool.map(nvi_stat, data_file_paths)
    
    #print("calculating ADF statistic")
    #writePath = "%s_analysis/ADFvalues.txt" % (inp)
    #txt_out = open(writePath, 'w')
    #txt_out.close
    #with multiprocessing.Pool(processes=1) as pool: # Use os.cpu_count() for max processes
    #    pool.map(adf_stat, sound_file_paths)
            
    ###################    
    print("collecting data")
    coll_data()
    print("normalizing data")
    norm_data()
    print("\nsignal analysis is complete\n")   
    
        
###############################################################
if __name__ == '__main__':
    main()
    
    
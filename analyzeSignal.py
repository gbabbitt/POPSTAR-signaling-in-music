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
from sklearn.preprocessing import MinMaxScaler
from hurst import compute_Hc, random_walk
import EntropyHub as eh
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
#num_cores = 1 # activate this line for identifying/removing files that stop script with errors

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
    if(header == "spch"):
        spchOpt = value
        print("speech-normalize",spchOpt)
    if(header == "self"):
        selfOpt = value
        print("self-normalize",selfOpt)
    if(header == "musi"):
        musiOpt = value
        print("music-normalize",musiOpt)
    if(header == "metro"):
        met = value
        print("my metronome option is",met)    
infile.close()
 ###### variable assignments ######
inp = ""+name+""
tm = int(tm)
fof = ""+fof+""
spchOpt = ""+spchOpt+""
selfOpt = ""+selfOpt+""
musiOpt = ""+musiOpt+""
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
        for filename in os.listdir(folder_path2):
            file_path = os.path.join(folder_path2, "%s" % (filename))
            #print(file_path)
            if os.path.isfile(file_path) and file_path[-4:] == ".wav":
                print("generating sound file list for %s%s" % (folder_path2,filename))
                sound_file_paths.append(file_path)
        for filename in os.listdir(folder_path2):
            file_path = os.path.join(folder_path2, "%s" % (filename))
            #print(file_path)
            if os.path.isfile(file_path) and file_path[-4:] == ".dat":
                print("generating data file list for %s%s" % (folder_path2,filename))
                data_file_paths.append(file_path)
        #print(sound_file_paths)
        #print(data_file_paths)
    return sound_file_paths
    return data_file_paths
        
########################################################################

# surprise metrics
def nvi_stat(item):
    path_objs = item.split("/")
    if(fileORfolder == "file"):
        filename = path_objs[2]
        writePath = "%s_analysis/NVIvalues.txt" % (inp)
    if(fileORfolder == "folder"):
        foldername = path_objs[2]
        filename = path_objs[3]     
        writePath = "%s_analysis/NVIvalues_%s.txt" % (inp,foldername)
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
    if(fileORfolder == "file"):
        print("NVI (note variability index) = %s for %s" % (NVI,filename))
    if(fileORfolder == "folder"):
        print("NVI (note variability index) = %s for %s in %s" % (NVI,filename,foldername))
    txt_out.write("%s,%s\n" % (filename,NVI))
    txt_out.close

def lzc_stat(item):
    path_objs = item.split("/")
    if(fileORfolder == "file"):
        filename = path_objs[2]
        writePath = "%s_analysis/LZCvalues.txt" % (inp)
    if(fileORfolder == "folder"):
        foldername = path_objs[2]
        filename = path_objs[3]     
        writePath = "%s_analysis/LZCvalues_%s.txt" % (inp,foldername) 
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
    if(fileORfolder == "file"):
        print("LZC (Lempel-Ziv complexity) = %s for %s" % (LZC,filename))
    if(fileORfolder == "folder"):
        print("LZC (Lempel-Ziv complexity) = %s for %s in %s" % (LZC,filename,foldername))
    txt_out.write("%s,%s\n" % (filename,LZC))
    txt_out.close
        

def adf_stat(item):
    path_objs = item.split("/")
    if(fileORfolder == "file"):
        filename = path_objs[2]
        writePath = "%s_analysis/ADFvalues.txt" % (inp)
    if(fileORfolder == "folder"):
        foldername = path_objs[2]
        filename = path_objs[3]     
        writePath = "%s_analysis/ADFvalues_%s.txt" % (inp,foldername)
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
    if(fileORfolder == "file"):
        print("ADF (augmented Dickey Fuller test) = %s for %s" % (ADFtestStat,filename))
    if(fileORfolder == "folder"):
        print("ADF (augmented Dickey Fuller test) = %s for %s in %s" % (ADFtestStat,filename,foldername))
    txt_out.write("%s,%s\n" % (filename,ADFtestStat))
    txt_out.close

def mse_stat(item):
    infile = item
    #print("myfile = %s" % item)    
    path_objs = item.split("/")
    if(fileORfolder == "file"):
        filename = path_objs[2]
        writePath = "%s_analysis/MSEvalues.txt" % (inp)
    if(fileORfolder == "folder"):
        foldername = path_objs[2]
        filename = path_objs[3]     
        writePath = "%s_analysis/MSEvalues_%s.txt" % (inp,foldername) 
    txt_out = open(writePath, 'a')
    
    samplingFrequency, signalData = wavfile.read(infile)
    #print(signalData[:,1])
    #print(samplingFrequency)
    # Matplotlib.pyplot.specgram() function to
    # generate spectrogram
    if(signalData.ndim != 1):
        #print("flattening signal to 1D")
        signalData = signalData[:,1]
    signalData = np.float32(signalData)
    #print(signalData)
    signalData = signalData[0:5000] # subset first N data points
    #signalData = np.random.choice(signalData, size=5000, replace=True) # subset random N data points
    # refined multiscale sample entropy using example code at
    # https://www.entropyhub.xyz/python/Examples/Ex7.html
    Mobj = eh.MSobject('SampEn', m = 4, r = 1.25)
    MSx, Ci = eh.rMSEn(signalData, Mobj, Scales = 5, F_Order = 3, F_Num = 0.6, RadNew = 4)
    if(fileORfolder == "file"):
        print("MSE (multiscale complexity index) = %s for %s" % (Ci,filename))
    if(fileORfolder == "folder"):
        print("MSE (multiscale complexity index) = %s for %s in %s" % (Ci,filename,foldername))
    txt_out.write("%s,%s\n" % (filename,Ci))
    txt_out.close
    
    
def mli_stat(item):
    infile = item
    print("myfile = %s" % item)    
    path_objs = item.split("/")
    if(fileORfolder == "file"):
        filename = path_objs[2]
        writePath = "%s_analysis/MLIvalues.txt" % (inp)
    if(fileORfolder == "folder"):
        foldername = path_objs[2]
        filename = path_objs[3]     
        writePath = "%s_analysis/MLIvalues_%s.txt" % (inp,foldername) 
    txt_out = open(writePath, 'a')
    
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
    if(isinstance(H, float)==True):
        mem_level = 1-(2*abs(H-0.5)) # rescale 0-1  
    if(isinstance(H, float)==False):
        mem_level = 1.0
    if(fileORfolder == "file"):
        print("MLI (inverse memory level index) = %s for %s" % (mem_level,filename))
    if(fileORfolder == "folder"):
        print("MLI (inverse memory level index) = %s for %s in %s" % (mem_level,filename,foldername))
    txt_out.write("%s,%s\n" % (filename,mem_level))
    txt_out.close
    
# control metrics
def f0_var_stat(item):
    path_objs = item.split("/")
    if(fileORfolder == "file"):
        filename = path_objs[2]
        writePath = "%s_analysis/FFVvalues.txt" % (inp)
    if(fileORfolder == "folder"):
        foldername = path_objs[2]
        filename = path_objs[3]     
        writePath = "%s_analysis/FFVvalues_%s.txt" % (inp,foldername)
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
    # equal tempered scale - octave range (octave 0 - 8)
    readPath = "NoteFreqHz.csv"
    note_freqs = pd.read_csv(readPath, delimiter=',',header=1)
    #print(note_freqs)
    for freq in f0:
        #print(freq)
        if(freq == "nan"):
            continue
        for i in range(len(note_freqs)-1):
            note = note_freqs.iloc[i,0]
            note_freq_lower = note_freqs.iloc[i-1,1]
            note_freq = note_freqs.iloc[i,1]
            note_freq_upper = note_freqs.iloc[i+1,1]
            octave = note_freqs.iloc[i,2]
            #print("%s %s %s" %(note, note_freq, octave))
            note_freq_upper_bound = note_freq + 0.5*(note_freq_upper - note_freq)
            note_freq_lower_bound = note_freq - 0.5*(note_freq - note_freq_lower)
            #print("bounds %s to %s" %(note_freq_lower_bound,note_freq_upper_bound))
            if(freq < note_freq_upper_bound and freq >= note_freq_lower_bound): 
                notes.append(note)
                #sum_diff = sum_diff + abs(freq - note_freq)
                sum_diff = sum_diff + (((freq - note_freq)**2)/note_freq)
                #print("detected %s in oct%s" % (note, octave))
        
    ##############
    #print("notes detected")
    #print(notes)
    n_notes = len(notes)
    if(sum_diff == 0):
        FFV = 0.000001
    if(sum_diff != 0):
        FFV = np.log(1/((sum_diff/(len(notes)))+0.000001))
        #FFV = 1/(sum_diff+0.000001)
    if(fileORfolder == "file"):
        print("FFV (f0 frequency control) = %s over %s notes for %s" % (FFV,n_notes,filename))
    if(fileORfolder == "folder"):
        print("FFV (f0 frequency control) = %s over %s notes for %s in %s" % (FFV,n_notes,filename,foldername))
    txt_out.write("%s,%s,%s\n" % (filename,FFV,n_notes))
    txt_out.close
    
    
def fn_levels_stat(item):
    path_objs = item.split("/")
    if(fileORfolder == "file"):
        filename = path_objs[2]
        writePath = "%s_analysis/HENvalues.txt" % (inp)
    if(fileORfolder == "folder"):
        foldername = path_objs[2]
        filename = path_objs[3]     
        writePath = "%s_analysis/HENvalues_%s.txt" % (inp,foldername) 
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
    if(np.sum(harmonic_energy) <= 1):
        HEN = 0
    if(np.sum(harmonic_energy) > 1):
        HEN = np.log(np.sum(harmonic_energy))
    if(fileORfolder == "file"):
        print("HEN (harmonic energy) = %s over for %s" % (HEN,filename))
    if(fileORfolder == "folder"):
        print("HEN (harmonic energy) = %s over for %s in %s" % (HEN,filename, foldername))
    txt_out.write("%s,%s\n" % (filename,HEN))
    txt_out.close

def beat_var(item):
    path_objs = item.split("/")
    if(fileORfolder == "file"):
        filename = path_objs[2]
        writePath = "%s_analysis/BIVvalues.txt" % (inp)
        writePath2 = "%s_analysis/EVIvalues.txt" % (inp)
    if(fileORfolder == "folder"):
        foldername = path_objs[2]
        filename = path_objs[3]     
        writePath = "%s_analysis/BIVvalues_%s.txt" % (inp,foldername)
        writePath2 = "%s_analysis/EVIvalues_%s.txt" % (inp,foldername)
    txt_out = open(writePath, 'a')
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
    n_beats = len(beat_intervals)
    mean_beat_interval = np.mean(beat_intervals)
    EVIsum = ((beat_intervals - mean_beat_interval)**2)/mean_beat_interval
    EVI = np.log(1/(np.sum(EVIsum)/(n_beats+0.000001)))
    BIV = np.log(1/((np.var(beat_intervals))+0.000001))
    # Print the estimated tempo and beat intervals
    #print(f"Estimated tempo: {tempo} BPM")
    #print(f"Beat intervals: {beat_intervals}")
    if(fileORfolder == "file"):
        print("BIV (1/beat interval deviation) = %s for %s" % (BIV,filename))
        print("EVI (1/beat interval sums of squares) = %s for %s" % (EVI,filename))
    if(fileORfolder == "folder"):
        print("BIV (1/beat interval deviation) = %s for %s in %s" % (BIV,filename,foldername))
        print("EVI (1/beat interval sums of squares) = %s for %s in %s" % (EVI,filename,foldername))
    txt_out.write("%s,%s\n" % (filename,BIV))
    txt_out.close
    txt_out2.write("%s,%s\n" % (filename,EVI))
    txt_out2.close

# energy metrics
def ampvar_stat(item):
    path_objs = item.split("/")
    if(fileORfolder == "file"):
        filename = path_objs[2]
        writePath = "%s_analysis/AMPvalues.txt" % (inp)
    if(fileORfolder == "folder"):
        foldername = path_objs[2]
        filename = path_objs[3]     
        writePath = "%s_analysis/AMPvalues_%s.txt" % (inp,foldername) 
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
    #AMP = np.log(np.var(signalData)) # amplitude variance
    AMP = np.log(np.sum(abs(signalData))) # amplitude volume
    if(fileORfolder == "file"):
        print("AMP (amplitude volume) = %s for %s" % (AMP,filename))
    if(fileORfolder == "folder"):
        print("AMP (amplitude volume) = %s for %s in %s" % (AMP,filename,foldername))
    txt_out.write("%s,%s\n" % (filename,AMP))
    txt_out.close
    
def dimension_stat(item):
    path_objs = item.split("/")
    if(fileORfolder == "file"):
        filename = path_objs[2]
        writePath = "%s_analysis/AC1values.txt" % (inp)
    if(fileORfolder == "folder"):
        foldername = path_objs[2]
        filename = path_objs[3] 
        writePath = "%s_analysis/AC1values_%s.txt" % (inp,foldername)
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
    if(fileORfolder == "file"):
        print("AC1 (1st order autocorrelation) = %s for %s" % (MAC,filename))
    if(fileORfolder == "folder"):
        print("AC1 (1st order autocorrelation) = %s for %s in %s" % (MAC,filename,foldername))
    txt_out.write("%s,%s\n" % (filename,MAC))
    txt_out.close
    
def tempo_stat(item):
    path_objs = item.split("/")
    if(fileORfolder == "file"):
        filename = path_objs[2]
        writePath = "%s_analysis/TEMPOvalues.txt" % (inp)
    if(fileORfolder == "folder"):
        foldername = path_objs[2]
        filename = path_objs[3]     
        writePath = "%s_analysis/TEMPOvalues_%s.txt" % (inp,foldername) 
    txt_out = open(writePath, 'a')
    infile = item
    song = AudioSegment.from_file(infile, format="wav") 
    #print(song.duration_seconds)
    dur = song.duration_seconds
    # Estimate the tempo (BPM)
    y, sr = librosa.load(item)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    if(isinstance(tempo, np.ndarray)==True):    
        tempo = tempo[0]
    if(fileORfolder == "file"):
        print("TEMPO (tempo - bpm) = %s for %s" % (tempo,filename))
    if(fileORfolder == "folder"):
        print("TEMPO (tempo - bpm) = %s for %s in %s" % (tempo,filename,foldername))
    txt_out.write("%s,%s\n" % (filename,tempo))
    txt_out.close
    
def coll_data():
    print("collecting data")
    writePath = "%s_analysis/features_raw.txt" % (inp)
    txt_out = open(writePath, 'w')
    txt_out.write("file,AC1values,AMPvalues,BIVvalues,EVIvalues,FFVvalues,HENvalues,LZCvalues,MSEvalues,NVIvalues,TEMPOvalues\n")
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
    readPath8 = "%s_analysis/MSEvalues.txt" % (inp)
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
    MSE_lines = txt_in8.readlines()
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
        for line in MSE_lines:
            line_split1 = line.split("_")
            seg_num = int(line_split1[0])
            line_split2 = line.split(",")
            if(i==seg_num):
                print("MLI matching %s to %s" % (i,seg_num))
                file_name = line_split2[0]
                MSE= float(line_split2[1])
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
                
        txt_out.write("%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" % (file_name,AC1,AMP,BIV,EVI,FFV,HEN,LZC,MSE,NVI,TEMPO))
    txt_out.close


def norm_data():
    print("normalizing data")
    readPath = "%s_analysis/features_raw.txt" % (inp)
    writePath = "%s_analysis/features_norm.txt" % (inp)
    writePath2 = "%s_analysis/ternary.txt" % (inp)
    writePath3 = "%s_analysis/ternary_norm.txt" % (inp)
    df = pd.read_csv(readPath, delimiter=',',header=0)
    #print(df)
    df = df.iloc[:, 1:] # drop newlines
    print(df)
    df[np.isinf(df)] = np.nan # replace inf with nan
    #df = df.drop(df.index[-1]) # drop last line due to lack of signal
    #df = pd.concat([df, df.iloc[[-1]]], ignore_index=True)  # copy last line to maintain proper index size
    ##############################################################
    ###### minmax normalization individually on each column ######
    ##############################################################
    if(selfOpt == "yes"):
        df_norm = df.copy()
        column = 'AC1values'
        df_norm[column] = MinMaxScaler().fit_transform(np.array(df_norm[column]).reshape(-1,1))
        column = 'AMPvalues'
        df_norm[column] = MinMaxScaler().fit_transform(np.array(df_norm[column]).reshape(-1,1))
        column = 'BIVvalues'
        df_norm[column] = MinMaxScaler().fit_transform(np.array(df_norm[column]).reshape(-1,1))
        column = 'EVIvalues'
        df_norm[column] = MinMaxScaler().fit_transform(np.array(df_norm[column]).reshape(-1,1))
        column = 'FFVvalues'
        df_norm[column] = MinMaxScaler().fit_transform(np.array(df_norm[column]).reshape(-1,1))
        column = 'HENvalues'
        df_norm[column] = MinMaxScaler().fit_transform(np.array(df_norm[column]).reshape(-1,1))
        column = 'LZCvalues'
        df_norm[column] = MinMaxScaler().fit_transform(np.array(df_norm[column]).reshape(-1,1))
        column = 'MSEvalues'
        df_norm[column] = MinMaxScaler().fit_transform(np.array(df_norm[column]).reshape(-1,1))
        column = 'NVIvalues'
        df_norm[column] = MinMaxScaler().fit_transform(np.array(df_norm[column]).reshape(-1,1))
        column = 'TEMPOvalues'
        df_norm[column] = MinMaxScaler().fit_transform(np.array(df_norm[column]).reshape(-1,1))
        df_norm = df_norm.fillna(0.000001) # replace nan and inf with near zero values
        print(df_norm)
           
    ########################################################
    ##### z-score to normalize signal to human speech  #####
    ########################################################
    if(spchOpt == "yes"):  # note: z score is rescaled from -1,1 to 0,1
        df_norm = df.copy() 
        sf = 0.5  # scaling factor
        column = 'AC1values'
        mean = 0.949214173352932
        sd = 0.0601940546883669
        df_norm[column] = np.array((((df_norm[column]-mean)/sd)*sf+1)/2)
        column = 'AMPvalues'
        mean =  23.4233734767798 
        sd = 5.61137334598524
        df_norm[column] = np.array((((df_norm[column]-mean)/sd)*sf+1)/2)
        column = 'BIVvalues'
        mean = 6.30536719531844  
        sd = 1.90448547602485
        df_norm[column] = np.array((((df_norm[column]-mean)/sd)*sf+1)/2)
        column = 'EVIvalues'
        mean = 6.80084850562953
        sd = 7.43177197236326
        df_norm[column] = np.array((((df_norm[column]-mean)/sd)*sf+1)/2)
        column = 'FFVvalues'
        mean = 2.9199533843274  
        sd = 0.719197572142818
        df_norm[column] = np.array((((df_norm[column]-mean)/sd)*sf+1)/2)
        column = 'HENvalues'
        mean = 8.42027318443921  
        sd = 1.83382917348368
        df_norm[column] = np.array((((df_norm[column]-mean)/sd)*sf+1)/2)
        column = 'LZCvalues'
        mean = 12.5059093945709  
        sd = 0.656758193570298
        df_norm[column] = np.array((((df_norm[column]-mean)/sd)*sf+1)/2)
        column = 'MSEvalues'
        mean = 0.959080514957559  
        sd = 0.569137483747565
        df_norm[column] = np.array((((df_norm[column]-mean)/sd)*sf+1)/2)
        column = 'NVIvalues'
        mean = 0.683926081755977  
        sd = 0.122670730550913
        df_norm[column] = np.array((((df_norm[column]-mean)/sd)*sf+1)/2)
        column = 'TEMPOvalues'
        mean = 124.785422049581  
        sd = 36.7788654827943
        df_norm[column] = np.array((((df_norm[column]-mean)/sd)*sf+1)/2)
        df_norm = df_norm.fillna(0.000001) # replace nan and inf with near zero values
        print(df_norm)
    ########################################################
    ##### z-score to normalize signal to human music  #####
    ########################################################
    if(musiOpt == "yes"):  # note: z score is rescaled from -1,1 to 0,1
        df_norm = df.copy() 
        sf = 0.5  # scaling factor
        column = 'AC1values'
        mean = 0.949214173352932
        sd = 0.0601940546883669
        df_norm[column] = np.array((((df_norm[column]-mean)/sd)*sf+1)/2)
        column = 'AMPvalues'
        mean =  23.4233734767798 
        sd = 5.61137334598524
        df_norm[column] = np.array((((df_norm[column]-mean)/sd)*sf+1)/2)
        column = 'BIVvalues'
        mean = 6.30536719531844  
        sd = 1.90448547602485
        df_norm[column] = np.array((((df_norm[column]-mean)/sd)*sf+1)/2)
        column = 'EVIvalues'
        mean = 6.80084850562953
        sd = 7.43177197236326
        df_norm[column] = np.array((((df_norm[column]-mean)/sd)*sf+1)/2)
        column = 'FFVvalues'
        mean = 2.9199533843274  
        sd = 0.719197572142818
        df_norm[column] = np.array((((df_norm[column]-mean)/sd)*sf+1)/2)
        column = 'HENvalues'
        mean = 8.42027318443921  
        sd = 1.83382917348368
        df_norm[column] = np.array((((df_norm[column]-mean)/sd)*sf+1)/2)
        column = 'LZCvalues'
        mean = 12.5059093945709  
        sd = 0.656758193570298
        df_norm[column] = np.array((((df_norm[column]-mean)/sd)*sf+1)/2)
        column = 'MSEvalues'
        mean = 0.959080514957559  
        sd = 0.569137483747565
        df_norm[column] = np.array((((df_norm[column]-mean)/sd)*sf+1)/2)
        column = 'NVIvalues'
        mean = 0.683926081755977  
        sd = 0.122670730550913
        df_norm[column] = np.array((((df_norm[column]-mean)/sd)*sf+1)/2)
        column = 'TEMPOvalues'
        mean = 124.785422049581  
        sd = 36.7788654827943
        df_norm[column] = np.array((((df_norm[column]-mean)/sd)*sf+1)/2)
        df_norm = df_norm.fillna(0.000001) # replace nan and inf with near zero values
        print(df_norm)
        
    with open(writePath, 'w') as txt_out:
        txt_out.write("AC1values,AMPvalues,BIVvalues,EVIvalues,FFVvalues,HENvalues,LZCvalues,MSEvalues,NVIvalues,TEMPOvalues\n")
        for index, row in df_norm.iterrows():
            line = ','.join(str("{:.8f}".format(x)) for x in row.values)  # Convert row to comma-separated string
            txt_out.write(line + '\n')  # Write line to file with newline character
        txt_out.close
    df_energy = df_norm[['AC1values', 'AMPvalues', 'TEMPOvalues']].mean(axis=1)
    df_control = df_norm[['FFVvalues', 'EVIvalues', 'HENvalues']].mean(axis=1)
    df_surprise = df_norm[['LZCvalues', 'MSEvalues', 'NVIvalues']].mean(axis=1)
    #df_energy = df_norm[['AC1values', 'AMPvalues', 'TEMPOvalues']].max(axis=1)
    #df_control = df_norm[['FFVvalues', 'EVIvalues', 'HENvalues']].max(axis=1)
    #df_surprise = df_norm[['LZCvalues', 'MSEvalues', 'NVIvalues']].max(axis=1)
    df_ternary = pd.concat([df_energy, df_control, df_surprise], axis=1)
    print(df_ternary)
    with open(writePath2, 'w') as txt_out:
        txt_out.write("energy,control,surprise\n")
        for index, row in df_ternary.iterrows():
            line = ','.join(str("{:.8f}".format(x)) for x in row.values)  # Convert row to comma-separated string
            txt_out.write(line + '\n')  # Write line to file with newline character
        txt_out.close
    df_ternary = df_ternary.abs() # hardens against extreme values in plots normalized to human speech
    df_ternary_norm = df_ternary.div(df_ternary.sum(axis=1), axis=0)
    print(df_ternary_norm)
    with open(writePath3, 'w') as txt_out:
        txt_out.write("energy,control,surprise\n")
        for index, row in df_ternary_norm.iterrows():
            line = ','.join(str("{:.8f}".format(x)) for x in row.values)  # Convert row to comma-separated string
            txt_out.write(line + '\n')  # Write line to file with newline character
        txt_out.close
    
def coll_data_batch():
    folder_path1 = "%s_analysis/intervals/" % inp
    #print(folder_path1)
    for foldername in os.listdir(folder_path1):
        folder_path2 = os.path.join(folder_path1, "%s" % (foldername))
        print(folder_path2)
        path_array = folder_path2.split("/")
        foldername = path_array[2]
        print(foldername)
           
        print("collecting data")
        writePath = "%s_analysis/features_raw_%s.txt" % (inp,foldername)
        txt_out = open(writePath, 'w')
        txt_out.write("file,AC1values,AMPvalues,BIVvalues,EVIvalues,FFVvalues,HENvalues,LZCvalues,MSEvalues,NVIvalues,TEMPOvalues\n")
        readPath1 = "%s_analysis/AC1values_%s.txt" % (inp,foldername)
        txt_in1 = open(readPath1, 'r')
        readPath2 = "%s_analysis/AMPvalues_%s.txt" % (inp,foldername)
        txt_in2 = open(readPath2, 'r')
        readPath3 = "%s_analysis/BIVvalues_%s.txt" % (inp,foldername)
        txt_in3 = open(readPath3, 'r')
        readPath4 = "%s_analysis/EVIvalues_%s.txt" % (inp,foldername)
        txt_in4 = open(readPath4, 'r')
        readPath5 = "%s_analysis/FFVvalues_%s.txt" % (inp,foldername)
        txt_in5 = open(readPath5, 'r')
        readPath6 = "%s_analysis/HENvalues_%s.txt" % (inp,foldername)
        txt_in6 = open(readPath6, 'r')
        readPath7 = "%s_analysis/LZCvalues_%s.txt" % (inp,foldername)
        txt_in7 = open(readPath7, 'r')
        readPath8 = "%s_analysis/MSEvalues_%s.txt" % (inp,foldername)
        txt_in8 = open(readPath8, 'r')
        readPath9 = "%s_analysis/NVIvalues_%s.txt" % (inp,foldername)
        txt_in9 = open(readPath9, 'r')
        readPath10 = "%s_analysis/TEMPOvalues_%s.txt" % (inp,foldername)
        txt_in10 = open(readPath10, 'r')
        AC1_lines = txt_in1.readlines()
        AMP_lines = txt_in2.readlines()
        BIV_lines = txt_in3.readlines()
        EVI_lines = txt_in4.readlines()
        FFV_lines = txt_in5.readlines()
        HEN_lines = txt_in6.readlines()
        LZC_lines = txt_in7.readlines()
        MSE_lines = txt_in8.readlines()
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
            for line in MSE_lines:
                line_split1 = line.split("_")
                seg_num = int(line_split1[0])
                line_split2 = line.split(",")
                if(i==seg_num):
                    print("MSE matching %s to %s" % (i,seg_num))
                    file_name = line_split2[0]
                    MSE= float(line_split2[1])
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
            
            txt_out.write("%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" % (file_name,AC1,AMP,BIV,EVI,FFV,HEN,LZC,MSE,NVI,TEMPO))
        txt_out.close


def norm_data_batch():
    folder_path1 = "%s_analysis/intervals/" % inp
    #print(folder_path1)
    for foldername in os.listdir(folder_path1):
        folder_path2 = os.path.join(folder_path1, "%s" % (foldername))
        print(folder_path2)
                    
        print("normalizing data")
        readPath = "%s_analysis/features_raw_%s.txt" % (inp,foldername)
        writePath = "%s_analysis/features_norm_%s.txt" % (inp,foldername)
        writePath2 = "%s_analysis/ternary_%s.txt" % (inp,foldername)
        writePath3 = "%s_analysis/ternary_norm_%s.txt" % (inp,foldername)
        df = pd.read_csv(readPath, delimiter=',',header=0)
        df = df.fillna(0.000001) # replace nan and inf with near zero values
        print(df)
        df = df.iloc[:, 1:]
        df[np.isinf(df)] = np.nan # replace inf with nan
        #df = df.drop(df.index[-1]) # drop last line due to lack of signal
        #df = pd.concat([df, df.iloc[[-1]]], ignore_index=True)  # copy last line to maintain proper index size
        ##############################################################
        ###### minmax normalization individually on each column ######
        ##############################################################
        if(selfOpt == "yes"):
            df_norm = df.copy()
            column = 'AC1values'
            df_norm[column] = MinMaxScaler().fit_transform(np.array(df_norm[column]).reshape(-1,1))
            column = 'AMPvalues'
            df_norm[column] = MinMaxScaler().fit_transform(np.array(df_norm[column]).reshape(-1,1))
            column = 'BIVvalues'
            df_norm[column] = MinMaxScaler().fit_transform(np.array(df_norm[column]).reshape(-1,1))
            column = 'EVIvalues'
            df_norm[column] = MinMaxScaler().fit_transform(np.array(df_norm[column]).reshape(-1,1))
            column = 'FFVvalues'
            df_norm[column] = MinMaxScaler().fit_transform(np.array(df_norm[column]).reshape(-1,1))
            column = 'HENvalues'
            df_norm[column] = MinMaxScaler().fit_transform(np.array(df_norm[column]).reshape(-1,1))
            column = 'LZCvalues'
            df_norm[column] = MinMaxScaler().fit_transform(np.array(df_norm[column]).reshape(-1,1))
            column = 'MSEvalues'
            df_norm[column] = MinMaxScaler().fit_transform(np.array(df_norm[column]).reshape(-1,1))
            column = 'NVIvalues'
            df_norm[column] = MinMaxScaler().fit_transform(np.array(df_norm[column]).reshape(-1,1))
            column = 'TEMPOvalues'
            df_norm[column] = MinMaxScaler().fit_transform(np.array(df_norm[column]).reshape(-1,1))
            df_norm = df_norm.fillna(0.000001) # replace nan and inf with near zero values
            print(df_norm)
           
        ########################################################
        ##### z-score to normalize signal to human speech  #####
        ########################################################
        if(spchOpt == "yes"):  # note: z score is rescaled from -1,1 to 0,1
            df_norm = df.copy() 
            sf = 0.5  # scaling factor
            column = 'AC1values'
            mean = 0.949214173352932
            sd = 0.0601940546883669
            df_norm[column] = np.array((((df_norm[column]-mean)/sd)*sf+1)/2)
            column = 'AMPvalues'
            mean =  23.4233734767798 
            sd = 5.61137334598524
            df_norm[column] = np.array((((df_norm[column]-mean)/sd)*sf+1)/2)
            column = 'BIVvalues'
            mean = 6.30536719531844  
            sd = 1.90448547602485
            df_norm[column] = np.array((((df_norm[column]-mean)/sd)*sf+1)/2)
            column = 'EVIvalues'
            mean = 6.80084850562953
            sd = 7.43177197236326
            df_norm[column] = np.array((((df_norm[column]-mean)/sd)*sf+1)/2)
            column = 'FFVvalues'
            mean = 2.9199533843274  
            sd = 0.719197572142818
            df_norm[column] = np.array((((df_norm[column]-mean)/sd)*sf+1)/2)
            column = 'HENvalues'
            mean = 8.42027318443921  
            sd = 1.83382917348368
            df_norm[column] = np.array((((df_norm[column]-mean)/sd)*sf+1)/2)
            column = 'LZCvalues'
            mean = 12.5059093945709  
            sd = 0.656758193570298
            df_norm[column] = np.array((((df_norm[column]-mean)/sd)*sf+1)/2)
            column = 'MSEvalues'
            mean = 0.959080514957559  
            sd = 0.569137483747565
            df_norm[column] = np.array((((df_norm[column]-mean)/sd)*sf+1)/2)
            column = 'NVIvalues'
            mean = 0.683926081755977  
            sd = 0.122670730550913
            df_norm[column] = np.array((((df_norm[column]-mean)/sd)*sf+1)/2)
            column = 'TEMPOvalues'
            mean = 124.785422049581  
            sd = 36.7788654827943
            df_norm[column] = np.array((((df_norm[column]-mean)/sd)*sf+1)/2)
            df_norm = df_norm.fillna(0.000001) # replace nan and inf with near zero values
            print(df_norm)
        ########################################################
        ##### z-score to normalize signal to human music  #####
        ########################################################
        if(musiOpt == "yes"):  # note: z score is rescaled from -1,1 to 0,1
            df_norm = df.copy() 
            sf = 0.5  # scaling factor
            column = 'AC1values'
            mean = 0.949214173352932
            sd = 0.0601940546883669
            df_norm[column] = np.array((((df_norm[column]-mean)/sd)*sf+1)/2)
            column = 'AMPvalues'
            mean =  23.4233734767798 
            sd = 5.61137334598524
            df_norm[column] = np.array((((df_norm[column]-mean)/sd)*sf+1)/2)
            column = 'BIVvalues'
            mean = 6.30536719531844  
            sd = 1.90448547602485
            df_norm[column] = np.array((((df_norm[column]-mean)/sd)*sf+1)/2)
            column = 'EVIvalues'
            mean = 6.80084850562953
            sd = 7.43177197236326
            df_norm[column] = np.array((((df_norm[column]-mean)/sd)*sf+1)/2)
            column = 'FFVvalues'
            mean = 2.9199533843274  
            sd = 0.719197572142818
            df_norm[column] = np.array((((df_norm[column]-mean)/sd)*sf+1)/2)
            column = 'HENvalues'
            mean = 8.42027318443921  
            sd = 1.83382917348368
            df_norm[column] = np.array((((df_norm[column]-mean)/sd)*sf+1)/2)
            column = 'LZCvalues'
            mean = 12.5059093945709  
            sd = 0.656758193570298
            df_norm[column] = np.array((((df_norm[column]-mean)/sd)*sf+1)/2)
            column = 'MSEvalues'
            mean = 0.959080514957559  
            sd = 0.569137483747565
            df_norm[column] = np.array((((df_norm[column]-mean)/sd)*sf+1)/2)
            column = 'NVIvalues'
            mean = 0.683926081755977  
            sd = 0.122670730550913
            df_norm[column] = np.array((((df_norm[column]-mean)/sd)*sf+1)/2)
            column = 'TEMPOvalues'
            mean = 124.785422049581  
            sd = 36.7788654827943
            df_norm[column] = np.array((((df_norm[column]-mean)/sd)*sf+1)/2)
            df_norm = df_norm.fillna(0.000001) # replace nan and inf with near zero values
            print(df_norm)
            
        with open(writePath, 'w') as txt_out:
            txt_out.write("AC1values,AMPvalues,BIVvalues,EVIvalues,FFVvalues,HENvalues,LZCvalues,MSEvalues,NVIvalues,TEMPOvalues\n")
            for index, row in df_norm.iterrows():
                line = ','.join(str("{:.8f}".format(x)) for x in row.values)  # Convert row to comma-separated string
                txt_out.write(line + '\n')  # Write line to file with newline character
            txt_out.close
        df_energy = df_norm[['AC1values', 'AMPvalues', 'TEMPOvalues']].mean(axis=1)
        df_control = df_norm[['FFVvalues', 'EVIvalues', 'HENvalues']].mean(axis=1)
        df_surprise = df_norm[['LZCvalues', 'MSEvalues', 'NVIvalues']].mean(axis=1)
        #df_energy = df_norm[['AC1values', 'AMPvalues', 'TEMPOvalues']].max(axis=1)
        #df_control = df_norm[['FFVvalues', 'EVIvalues', 'HENvalues']].max(axis=1)
        #df_surprise = df_norm[['LZCvalues', 'MSEvalues', 'NVIvalues']].max(axis=1)
        df_ternary = pd.concat([df_energy, df_control, df_surprise], axis=1)
        print(df_ternary)
        with open(writePath2, 'w') as txt_out:
            txt_out.write("energy,control,surprise\n")
            for index, row in df_ternary.iterrows():
                line = ','.join(str("{:.8f}".format(x)) for x in row.values)  # Convert row to comma-separated string
                txt_out.write(line + '\n')  # Write line to file with newline character
            txt_out.close
        df_ternary = df_ternary.abs() # hardens against extreme values in plots normalized to human speech
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
    with multiprocessing.Pool(processes=num_cores) as pool: # Use os.cpu_count() for max processes
        pool.map(ampvar_stat, sound_file_paths)
    print("calculating size/dimension")
    with multiprocessing.Pool(processes=num_cores) as pool: # Use os.cpu_count() for max processes
        pool.map(dimension_stat, sound_file_paths)
    print("calculating local tempo")
    with multiprocessing.Pool(processes=num_cores) as pool: # Use os.cpu_count() for max processes
        pool.map(tempo_stat, sound_file_paths)
    
    ####################
    # control metrics
    ####################
    print("calculating f0 control")
    with multiprocessing.Pool(processes=num_cores) as pool: # Use os.cpu_count() for max processes
        pool.map(f0_var_stat, sound_file_paths)
    print("calculating fn harmonic energy")
    with multiprocessing.Pool(processes=num_cores) as pool: # Use os.cpu_count() for max processes
        pool.map(fn_levels_stat, sound_file_paths)
    print("calculating beat interval control")
    with multiprocessing.Pool(processes=num_cores) as pool: # Use os.cpu_count() for max processes
        pool.map(beat_var, sound_file_paths)
            
    ####################
    # surprise metrics
    ####################
    print("calculating Lempel-Ziv complexity")
    with multiprocessing.Pool(processes=num_cores) as pool: # Use os.cpu_count() for max processes
        pool.map(lzc_stat, sound_file_paths)
    print("calculating MSE statistic")
    with multiprocessing.Pool(processes=num_cores) as pool: # Use os.cpu_count() for max processes
        pool.map(mse_stat, sound_file_paths)
    print("calculating NVI statistic (zero order)")
    print("(Sawant et al. 2021 in MEE-BES)")
    with multiprocessing.Pool(processes=num_cores) as pool: # Use os.cpu_count() for max processes
        pool.map(nvi_stat, data_file_paths)
    
    #print("calculating ADF statistic")
    #writePath = "%s_analysis/ADFvalues.txt" % (inp)
    #txt_out = open(writePath, 'w')
    #txt_out.close
    #with multiprocessing.Pool(processes=1) as pool: # Use os.cpu_count() for max processes
    #    pool.map(adf_stat, sound_file_paths)
          
    ###################    
    if(fileORfolder == "file"):
        print("collecting data")
        coll_data()
        print("normalizing data")
        norm_data()
    if(fileORfolder == "folder"):
        print("collecting data")
        coll_data_batch()
        print("normalizing data")
        norm_data_batch()
    
    
    print("\nsignal analysis is complete\n")   
    
        
###############################################################
if __name__ == '__main__':
    main()
    
    
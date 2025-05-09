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
infile.close()
 ###### variable assignments ######
inp = ""+name+""
tm = int(tm)
fof = ""+fof+""
lyr = ""+lyr+""
maxOpt = ""+maxOpt+""
fileORfolder = fof

if(fof=="file"):
    # calculate number of faces for single file
    lst = os.listdir("%s_analysis/intervals/" % inp) # your directory path
    face_num = int(len(lst)/4)  # note folder has 4 types of files
    print("number of Chernoff faces is %s" % face_num)

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
    LZC = len(sub_strings)
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
    mem_level = 2*abs(H-0.5) # rescale 0-1
    print("MLI (memory level index) = %s for %s" % (mem_level,filename))
    txt_out.write("%s,%s\n" % (filename,mem_level))
    txt_out.close
    
# control metrics
def f0_var_stat():
    print("calculating f0 variance")
def f0_even_stat():
    print("calculating f0 evenness")    
def fn_levels_stat():
    print("calculating fn levels")

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
    AMP = np.var(signalData)
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
#################################################################################
####################  main program      #########################################
#################################################################################
def main():
    if(fileORfolder == "file"):
        create_file_lists()
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
        f0_var_stat()
        f0_even_stat()
        fn_levels_stat()
        
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
        '''
        print("calculating ADF statistic")
        writePath = "%s_analysis/ADFvalues.txt" % (inp)
        txt_out = open(writePath, 'w')
        txt_out.close
        with multiprocessing.Pool(processes=1) as pool: # Use os.cpu_count() for max processes
            pool.map(adf_stat, sound_file_paths)
        '''
        ###################    
        print("\nsignal analysis is complete\n")   
    if(fileORfolder == "folder"):
        analyze_signal_batch_list()
        with multiprocessing.Pool(processes=num_cores) as pool: # Use os.cpu_count() for max processes
            pool.map(analyze_signal_batch, file_paths)
        print("\nsignal analysis is complete\n")
###############################################################
if __name__ == '__main__':
    main()
    
    
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

import multiprocessing

# IMPORTANT NOTE - run in base conda env, not in atomdance conda env   
################################################################################
'''
inp0 = input("\nChoose 'full' or 'fast' analysis for NVI index (default = full)\n" )
inp00 = input("\nChoose 'trim' if you need to employ audio cutter (recommended for music files) (default = no)\n" )
inp = input("\nName of sound file OR batch folder to analyze (e.g. type 'myfile' NOT 'myfile.wav')\n" )
tm = input("\nEnter the time interval length to analyze (in seconds) (e.g. 30)\n" )

'''
inp0 = "full"
inp00 = "y"
inp = "test"
tm = 20 # interval length in seconds
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
 ###### variable assignments ######
inp = ""+name+""
tm = int(tm)

if not os.path.exists('%s_analysis' % inp):
        os.mkdir('%s_analysis' % inp)
if not os.path.exists('%s_analysis/intervals' % inp):
        os.mkdir('%s_analysis/intervals' % inp)        
################################################################################
#########################   sonogram generator  ################################
################################################################################
#infile= input("\nEnter path/name of input sound file (name.wav)\n")   
#outfile = input("\nEnter path/name of output image file (name.png)\n")
# IMPORTANT NOTE - run in base conda env, not in atomdance conda env  

input1 = "%s.wav" % inp
input1alt = "%s.mp3" % inp
input2 = "%s.png" % inp
input3 = "%s.dat" % inp
input4 = "%s.txt" % inp
input5 = "%s.jpg" % inp
input6 = "%s_signal.jpg" % inp
input7 = "%s_shortMemoryBoost.wav" % inp
input8 = "%s_longMemoryBoost.wav" % inp
input9 = "%s_shortMemoryBoost.jpg" % inp
input10 = "%s_longMemoryBoost.jpg" % inp

if os.path.isfile(input1):
    print("user input is a .wav file")
    fileORfolder = "file"
    #inp2 = input("Do you want to activate bootstrapping? (y or n)\n")
elif os.path.isfile(input1alt):
    print("user input is a .mp3 file")
    print("converting to .wav format for %s" % inp) 
    song = AudioSegment.from_file(input1alt, format="mp3") 
    song.export(input1, format="wav")
    fileORfolder = "file"
    #inp2 = input("Do you want to activate bootstrapping? (y or n)\n")
elif os.path.isdir(inp):
    print("user input is a folder")
    fileORfolder = "folder"
  
else:
    print("Invalid Path")
    exit()
#################################################################################
####################  preprocessing     #########################################
#################################################################################
def trim_wav(): 
    print("trimming %s" % input1)
    # Open an mp3 file 
    song = AudioSegment.from_file(input1, format="wav") 
    # start and end time 
    if(inp00 == "trim"):
        start = 2000  # note 1000 = 1 second
        end = -2000
    else:
        start = 100  # note 1000 = 1 second
        end = -100
    # song clip of 10 seconds from starting 
    ftrim = song[start: end] 
    # save file 
    ftrim.export("%s_analysis/trimmed_%s" % (inp, input1), format="wav") 
    print("trimmed %s file is created and saved" % input1)

def trim_wav_batch(): 
    print("converting to .wav format for %s folder" % inp)
    print("trimming %s" % inp)
    lst = os.listdir(inp) # your directory path
    number_files = len(lst)
    print("number of files")
    print(number_files)
    dir_list = os.listdir(inp)
    print(dir_list)
    for i in range(number_files):    
        # Open an mp3 file 
        filename = dir_list[i]
        #print(filename)
        if os.path.isfile("%s/%s.mp3" % (inp,filename[:-4])):  # if .mp3
            song = AudioSegment.from_file("%s/%s" % (inp,filename), format="mp3") 
        else:  # else if .wav
            song = AudioSegment.from_file("%s/%s" % (inp,filename), format="wav") 
        # start and end time 
        if(inp00 == "trim"):
            start = 2000  # note 1000 = 1 second
            end = -2000
        else:
            start = 100  # note 1000 = 1 second
            end = -100
        # song clip of 10 seconds from starting 
        ftrim = song[start: end] 
        # save file 
        ftrim.export("%s_analysis/%s" % (inp,filename), format="wav") 
        print("trimmed %s is created and saved" % filename)

def time_sample(): 
    print("segmenting %s" % input1)
    print("length of file (seconds)")
    #tm = 20 # interval length in seconds
    song = AudioSegment.from_file("%s_analysis/trimmed_%s" % (inp,input1), format="wav") 
    print(song.duration_seconds)
    dur = song.duration_seconds
    ints = int(dur*4)-tm  # analyze in 1/4 second sliding window
    # start and end time 
    for i in range(ints): 
        start = i*250  # note 500 = 0.5 second
        end = i*250+tm*250
        print("start: %s end: %s" % (start,end))
        # song interval 
        finterval = song[start: end] 
        # save file 
        finterval.export("%s_analysis/intervals/%s_%s" % (inp, i, input1), format="wav") 
        print("interval %s for %s file is created and saved" % (i,input1))

def time_sample_batch(): 
    print("trimming %s" % inp)
    print("converting to .wav format for %s folder" % inp)
    print("trimming %s" % inp)
    lst = os.listdir(inp) # your directory path
    number_files = len(lst)
    print("number of files")
    print(number_files)
    dir_list = os.listdir(inp)
    print(dir_list)
    for i in range(number_files):    
        # Open an mp3 file 
        filename = dir_list[i]
        #print(filename)
        if not os.path.exists('%s_analysis/intervals/%s' % (inp,filename)):
            os.mkdir('%s_analysis/intervals/%s' % (inp,filename))
        #tm = 20 # interval length in seconds
        if os.path.isfile("%s/%s.mp3" % (inp,filename[:-4])):  # if .mp3
            song = AudioSegment.from_file("%s/%s" % (inp,filename), format="mp3") 
        else:  # else if .wav
            song = AudioSegment.from_file("%s/%s" % (inp,filename), format="wav")
        print(song.duration_seconds)
        dur = song.duration_seconds
        ints = int(dur*4)-tm  # analyze in 1/4 second sliding window
        # start and end time 
        for j in range(ints): 
            start = i*250  # note 500 = 0.5 second
            end = i*250+tm*250
            print("start: %s end: %s" % (start,end))
            # song interval 
            finterval = song[start: end] 
            # save file 
            finterval.export("%s_analysis/intervals/%s/%s_%s" % (inp, filename, j, inp), format="wav") 
            print("interval %s for %s file is created and saved" % (j, filename))   
            
def create_sonogram_list():   
    folder_path = "%s_analysis/intervals/" % inp
    print(folder_path)
    global file_paths
    file_paths = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, "%s" % (filename))
        #print(file_path)
        if os.path.isfile(file_path):
            print("generating sonograms for %s" % (filename))
            file_paths.append(file_path)
    print(file_paths)
    return file_paths
             
def create_sonogram(item):
            # Process the file
            infile = item
            infile_array = infile.split("/")
            infile_segment = infile_array[2]
            fname = infile_segment[:-4]
            print("generating sonograms for %s" % (fname))
            outfile = "%s_analysis/intervals/%s.png" % (inp,fname)
            outfile2 = "%s_analysis/intervals/%s_signal.jpg" % (inp,fname)
            samplingFrequency, signalData = wavfile.read(infile)
            #print(signalData)
            #print(signalData[:,1])
            if(signalData.ndim != 1):
                signalData = signalData[:,1]
        
            # generate signal plot (time domain)
            mySignal = signalData
            myIndices = []
            for k in range(len(mySignal)):
                myIndex = str(k+1)
                #print("myIndex %s" % myIndex)
                myIndices.append(myIndex)
            myIndices = np.array(myIndices)
            myIndices = myIndices[::100]
            mySignal = mySignal[::100]
            #print(len(mySignal))
            plt.plot(myIndices, mySignal, c = 'b')
            plt.xlabel("TIME")
            plt.ylabel("SIGNAL")
            plt.xticks([]) 
            plt.savefig(outfile2)
            plt.close()
    
            #print(samplingFrequency)
            # Matplotlib.pyplot.specgram() function to
   
    
            # generate spectrogram
            plt.specgram(signalData, Fs=samplingFrequency,NFFT=2048)
 
            # Set the title of the plot, xlabel and ylabel
            # and display using show() function
            plt.title("spectrogram for %s" % input1)
            plt.xlabel("TIME")
            plt.ylabel("FREQ")
            plt.savefig(outfile)
            plt.close()
            # export to txt
            ls=plt.specgram(signalData, Fs=samplingFrequency,NFFT=2048)
            #print(ls[0].shape)
            shp = ls[0].shape
            global n_cols
            if(inp0 == 'full'):
                n_cols = shp[1]
            if(inp0 == 'fast' and shp[1] >= 2000):
                n_cols = 2000
            if(inp0 == 'fast' and shp[1] < 2000):
                n_cols = shp[1]   
            #print("number of notes (i.e. columns)")
            #print(n_cols)
            if(inp0 == 'fast' and shp[1] >= 2000):
                print("...NVI will be limited to first 2000 cols")
            with open("%s_analysis/intervals/%s.dat" % (inp,fname), 'w') as ffile:
                for spectros in ls[0]:
                    for spectro in spectros:
                        spectro = round(spectro,4)
                        lline = "%s\t" % spectro
                        #print(lline)
                        ffile.write(lline)
                    # one row written 
                    ffile.write("\n")
                ffile.close
            plt.close()

def create_sonogram_batch_list(): 
    folder_path1 = "%s_analysis/intervals/" % inp
    #print(folder_path1)
    global file_paths
    file_paths = []
    for foldername in os.listdir(folder_path1):
        folder_path2 = os.path.join(folder_path1, "%s" % (foldername))
        print(folder_path2)
        # Process the file
        print("generating sonograms for %s" % folder_path2)
        global n_cols_array
        n_cols_array = []
        lst = os.listdir(folder_path2) # your directory path
        number_files = len(lst)
        print("number of files")
        print(number_files)
        dir_list = os.listdir(folder_path2)
        #print(dir_list)
        myIndices = []
        for filename in os.listdir(folder_path2):
            file_path = os.path.join(folder_path2, "%s" % (filename))
            #print(file_path)
            if os.path.isfile(file_path):
                print("generating sonograms for %s" % (filename))
                file_paths.append(file_path)
    print(file_paths)
    return file_paths

def create_sonogram_batch(item): 
                infile = item
                infile_array = infile.split("/")
                infile_segment = infile_array[3]
                fname = infile_segment
                folder = infile_array[2]
                print("generating sonograms for %s %s" % (folder,fname))
                outfile = "%s_analysis/intervals/%s/%s.png" % (inp,folder,fname)
                outfile2 = "%s_analysis/intervals/%s/%s_signal.jpg" % (inp,folder,fname)
                
                
                #infile = file_path
                #input2 = "%s.png" % filename
                #input6 = "%s_signal.jpg" % filename
                #outfile = "%s_analysis/intervals/%s/%s_%s" % (inp,foldername,itr,input2)
                #outfile2 = "%s_analysis/intervals/%s/%s_%s" % (inp,foldername,itr,input6)
                samplingFrequency, signalData = wavfile.read(infile)
                #print(infile)
                #print(signalData[:,1])
                #print(samplingFrequency)
                # Matplotlib.pyplot.specgram() function to
                if(signalData.ndim != 1):
                   signalData = signalData[:,1]
                # generate signal plot (time domain)
                mySignal = signalData
                myIndices = []
                for k in range(len(mySignal)):
                    myIndex = str(k+1)
                    myIndices.append(myIndex)
                myIndices = np.array(myIndices)
                #print(mySignal.ndim)
                #print(myIndices.ndim)
                plt.plot(myIndices, mySignal, c = 'b')
                plt.xlabel("TIME")
                plt.ylabel("SIGNAL")
                plt.xticks([]) 
                plt.savefig(outfile2)
                plt.close()
        
                # generate spectrogram
                #print(len(signalData))
                plt.specgram(signalData, Fs=samplingFrequency,NFFT=2048)
 
                # Set the title of the plot, xlabel and ylabel
                # and display using show() function
                plt.title("spectrogram for %s" % infile)
                plt.xlabel("TIME")
                plt.ylabel("FREQ")
                plt.savefig(outfile)
                plt.close()
                # export to txt
                ls=plt.specgram(signalData, Fs=samplingFrequency,NFFT=2048)
                #print(ls[0].shape)
                shp = ls[0].shape
                global n_cols
                if(inp0 == 'full'):
                    n_cols = shp[1]
                if(inp0 == 'fast' and shp[1] >= 2000):
                    n_cols = 2000
                if(inp0 == 'fast' and shp[1] < 2000):
                    n_cols = shp[1] 
                #print("generating sonogram for %s" % filename)
                #print("number of notes (i.e. columns)")
                #print(n_cols)
                if(inp0 == 'fast' and shp[1] >= 2000):
                    print("...NVI will be limited to first 2000 cols")
                n_cols_array.append(n_cols)
                #input3 = "%s.dat" % filename
                with open("%s_analysis/intervals/%s/%s.dat" % (inp,folder,fname), 'w') as ffile:
                    for spectros in ls[0]:
                        for spectro in spectros:
                            spectro = round(spectro,4)
                            lline = "%s\t" % spectro
                            #print(lline)
                            ffile.write(lline)
                        # one row written 
                        ffile.write("\n")
                    ffile.close
                plt.close()
            

        
#################################################################################
####################  main program      #########################################
#################################################################################
def main():
    if(fileORfolder == "file"):
        trim_wav()
        time_sample()
        create_sonogram_list()
        with multiprocessing.Pool(processes=num_cores) as pool: # Use os.cpu_count() for max processes
            pool.map(create_sonogram, file_paths)
        print("\nsound extractions are complete\n")    
    if(fileORfolder == "folder"):
        trim_wav_batch()
        time_sample_batch()
        create_sonogram_batch_list()
        with multiprocessing.Pool(processes=num_cores) as pool: # Use os.cpu_count() for max processes
            pool.map(create_sonogram_batch, file_paths)
        print("\nsound extractions are complete\n")
###############################################################
if __name__ == '__main__':
    main()
    
    
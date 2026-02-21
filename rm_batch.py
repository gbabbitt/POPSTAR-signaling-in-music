#!/usr/bin/env python

#############################################################################
######   POPSTAR software for detecting fitness signaling in music
######   produced by Dr. Gregory A. Babbitt
######   and students at the Rochester Instituteof Technology in 2025.
######   Offered freely without guarantee.  License under GPL v3.0
#############################################################################


import getopt, sys # Allows for command line arguments
import os
import shutil
import pandas as pd
import numpy as np
import scipy as sp
from pydub import AudioSegment
import soundfile
import librosa 
# IMPORTANT NOTE - run in base conda env, not in atomdance conda env   
################################################################################
in_dir = input("Enter folder name: ")

inp = in_dir

def time_sample_batch(): 
    
    lst = os.listdir(inp) # your directory path
    number_files = len(lst)
    print("number of files")
    print(number_files)
    dir_list = os.listdir(inp)
    print(dir_list)
    for i in range(number_files):    
        # Open an mp3 file 
        filename = dir_list[i]
        file_path = "%s/%s" % (inp,filename)
        #print(filename)
        
        #tm = 20 # interval length in seconds
        if os.path.isfile("%s/%s.mp3" % (inp,filename[:-4])):  # if .mp3
            song = AudioSegment.from_file("%s/%s" % (inp,filename), format="mp3") 
        else:  # else if .wav
            song = AudioSegment.from_file("%s/%s" % (inp,filename), format="wav")
        print(song.duration_seconds)
        dur = song.duration_seconds
        # Estimate the tempo (BPM)
        y, sr = librosa.load(file_path)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        print("tempo = %s" % str(tempo))
        if(tempo == 0):
            os.remove("%s/%s" %(inp,filename)) # Delete the file
            print("%s was removed from directory" % filename)
            continue
        tempo = tempo[0]
        total_beats = (dur/60*tempo)
        beat_int = dur/total_beats
        print("tempo = %s bpm" % tempo)
        print("beat interval = %s sec" % beat_int)
        print("total beats = %s" % total_beats)
         
       
def main():
    
    time_sample_batch()
    print("removed all files where TEMPO = 0 ...now complete")

###############################################################

    
if __name__ == '__main__':
    main()

    
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
import matplotlib.pyplot as plt
import yt_dlp
import noisereduce as nr
import soundfile as sf
import pyloudnorm as pyln
# IMPORTANT NOTE - run in base conda env, not in atomdance conda env   
################################################################################

if os.path.exists('YouTube_audio'):
    print("folder already exists...")
if not os.path.exists('YouTube_audio'):
        os.mkdir('YouTube_audio')

def move_wav_files(source_folder, destination_folder):
    """Moves all .wav files from source_folder to destination_folder."""

    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    for filename in os.listdir(source_folder):
        if filename.endswith(".wav"):
            source_path = os.path.join(source_folder, filename)
            destination_path = os.path.join(destination_folder, filename)
            shutil.move(source_path, destination_path)
            print(f"Moved: {filename}")

        
def main():
    ##################
    #video_url = input("Enter the video URL: ")
    # read txt file
    infile = open("urls.txt", "r")
    infile_lines = infile.readlines()
    for x in range(len(infile_lines)):
        infile_line = infile_lines[x]
        #print(infile_line)
        infile_line_array = str.split(infile_line, ",")
        myURL = infile_line_array[0]
        print("my URL is ",myURL)
        cmd = "yt-dlp -x --audio-format wav %s" % (myURL)
        os.system(cmd)
    infile.close()
    source_folder = "." # Replace with your source folder path
    destination_folder = "YouTube_audio" # Replace with your destination folder path
    move_wav_files(source_folder, destination_folder)
    # EBU R128 multi-track loudness normalization
    my_dir = "YouTube_audio"
    my_dir_new = "YouTube_norm"
    if not os.path.exists(my_dir_new):
        os.makedirs(my_dir_new)
    for filename in os.listdir(my_dir):
        file_path = os.path.join(my_dir, filename) # Construct full path
        file_path_new = os.path.join(my_dir_new, filename) # Construct full path
        print(file_path)
        print(file_path_new)
        if os.path.isfile(file_path): 
            print(f"Reading file: {file_path}")
            data, rate = sf.read(file_path)
            meter = pyln.Meter(rate) # Create a BS.1770 loudness meter
            loudness = meter.integrated_loudness(data) # Measure the integrated loudness of the audio
            loudness_normalized_audio = pyln.normalize.loudness(data, loudness, -15.0)# Loudness normalize audio to a target LUFS (e.g., -23 LUFS)
            sf.write(file_path_new, loudness_normalized_audio, rate)# Save the normalized audio
        
    print("Audio download complete")

###############################################################

    
if __name__ == '__main__':
    main()

    
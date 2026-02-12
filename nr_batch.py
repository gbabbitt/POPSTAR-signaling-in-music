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
in_dir = input("Enter folder name: ")
out_dir = "%s_norm" % in_dir

if os.path.exists(out_dir):
    print("folder already exists...")
if not os.path.exists(out_dir):
        os.mkdir(out_dir)

       
def main():
    
    # EBU R128 multi-track loudness normalization
    
    my_dir = in_dir
    my_dir_new = out_dir
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
            reduced_noise_audio = nr.reduce_noise(y=loudness_normalized_audio, sr=rate)
            #sf.write(file_path_new, loudness_normalized_audio, rate)# Save the normalized audio
            sf.write(file_path_new, reduced_noise_audio, rate)# Save the normalized audio
    print("Audio normalization and denoising complete")

###############################################################

    
if __name__ == '__main__':
    main()

    
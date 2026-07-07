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
#import noisereduce as nr
#import soundfile as sf
#import pyloudnorm as pyln
# IMPORTANT NOTE - run in base conda env, not in atomdance conda env   
################################################################################

if os.path.exists('YouTube_audio'):
    print("folder already exists...")
if not os.path.exists('YouTube_video'):
        os.mkdir('YouTube_video')
if not os.path.exists('YouTube_video_final'):
        os.mkdir('YouTube_video_final')        

def repair_files(source_folder, destination_folder):
    
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    for filename in os.listdir(source_folder):
        if filename.endswith(".mp4"):
            # convert to H.264 mp4 to avoid hardware incompatible code (i.e gpu accelerated code of av1 codec)
            cmd = "ffmpeg -i %s/%s -c:v libx264 -crf 23 -c:a copy %s/repaired_%s" % (source_folder,filename,destination_folder,filename)
            os.system(cmd)
            print(f"repaired: {filename}")

def move_files(source_folder, destination_folder):
    """Moves all .mp4 files from source_folder to destination_folder."""

    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    count = 1
    for filename in os.listdir(source_folder):
        if filename.endswith(".mp4"):
            newname = "yt%s.mp4" % count
            source_path = os.path.join(source_folder, filename)
            destination_path = os.path.join(destination_folder, newname)
            shutil.move(source_path, destination_path)
            count = count + 1
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
        cmd = "yt-dlp -f bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4] --merge-output-format mp4 %s" % (myURL)
        #cmd = "yt-dlp -f bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4] mp4 %s" % (myURL)
        #cmd = "yt-dlp -x --audio-format wav %s" % (myURL)
        os.system(cmd)
    infile.close()
    ###################
    source_folder = "." 
    destination_folder = "YouTube_video" 
    move_files(source_folder, destination_folder)
    ###################
    source_folder = "YouTube_video" 
    destination_folder = "YouTube_video_final" 
    repair_files(source_folder, destination_folder)
    print("Video download complete")

###############################################################

    
if __name__ == '__main__':
    main()

    
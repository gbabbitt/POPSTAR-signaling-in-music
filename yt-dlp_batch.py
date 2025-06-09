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
import yt_dlp

# IMPORTANT NOTE - run in base conda env, not in atomdance conda env   
################################################################################

if os.path.exists('YouTube_audio'):
    print("folder already exists...")
if not os.path.exists('YouTube_audio'):
        os.mkdir('YouTube_audio')
if os.path.exists('YouTube_video'):
    print("folder already exists...")
if not os.path.exists('YouTube_video'):
        os.mkdir('YouTube_video')
        
def download_audio(url, output_path='YouTube_audio', output_format='mp3'):
       ydl_opts = {
           'format': 'bestaudio/best',
           'outtmpl': f'{output_path}/%(title)s.%(ext)s',
           'extract_audio': True,
           'audio_format': output_format,
           'postprocessor_args': ['-vn'] 
       }
       with yt_dlp.YoutubeDL(ydl_opts) as ydl:
           ydl.download([url])

def download_video(url, output_path='YouTube_video', output_format='mp4'):
       ydl_opts = {
           'format': 'bestvideo/best',
           'outtmpl': f'{output_path}/%(title)s.%(ext)s',
           'extract_audio': False,
           'audio_format': output_format,
           'postprocessor_args': ['-vn'] 
       }
       with yt_dlp.YoutubeDL(ydl_opts) as ydl:
           ydl.download([url])
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
        download_audio(myURL)
        download_video(myURL)
    infile.close()
    ##################
    folder_path = "YouTube_audio"
    old_extension = ".webm"
    new_extension = ".mp3"
    for filename in os.listdir(folder_path):
        if filename.endswith(old_extension):
            new_name = filename.replace(old_extension, new_extension)
            os.rename(os.path.join(folder_path, filename), os.path.join(folder_path, new_name))
    print("Audio download complete")

###############################################################

    
if __name__ == '__main__':
    main()

    
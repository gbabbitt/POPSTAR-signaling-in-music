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
import cv2
from moviepy.editor import VideoFileClip
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
if not os.path.exists('YouTube_video_noBackground'):
        os.mkdir('YouTube_video_noBackground')
if not os.path.exists('YouTube_video_noBackground_audio'):
        os.mkdir('YouTube_video_noBackground_audio')
        
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
            #newname = "yt%s.mp4" % count
            source_path = os.path.join(source_folder, filename)
            newname = filename.replace(" ", "") # remove whitespace
            destination_path = os.path.join(destination_folder, newname)
            shutil.move(source_path, destination_path)
            count = count + 1
            print(f"Moved: {filename}")
    
def remove_background(source_folder, destination_folder, destination_folder2):
    
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    if not os.path.exists(destination_folder2):
        os.makedirs(destination_folder2)   
        
    count = 1
    for filename in os.listdir(source_folder):
        if filename.endswith(".mp4"):
            print("removing background from %s" % filename)
            #newname = "yt_filter%s.mp4" % count
            #newname2 = "yt_filter_audio%s.mp4" % count
            source_path = os.path.join(source_folder, filename)
            destination_path = os.path.join(destination_folder, filename)
            destination_path2 = os.path.join(destination_folder2, filename)
            # Open the video file (or use 0 for webcam)
            cap = cv2.VideoCapture(source_path)
            if not cap.isOpened():
                print('Error opening video file')
                exit()
            # Get video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            # Create the MOG2 background subtractor object
            # detectShadows=True marks shadows in gray; set to False if not needed
            fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
            # Setup VideoWriter to save the grayscale mask (isColor=False)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(destination_path, fourcc, fps, (width, height), isColor=False)

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Apply the background subtractor to get the foreground mask
                fgmask = fgbg.apply(frame)
                # Write mask frame to output video
                out.write(fgmask)
                '''
                # OPTIONAL : Display the original frame and the processed mask
                cv2.imshow('Original Frame', frame)
                cv2.imshow('Foreground Mask', fgmask)
                '''
                # Press 'Esc' key to exit the loop
                if cv2.waitKey(30) & 0xFF == 27:
                    break

            # Release video resources and close windows
            cap.release()
            out.release()
            cv2.destroyAllWindows()
            print(f"removed background: {filename}")
            
            # 4. Re-integrate original audio using MoviePy
            print("Merging original audio track into the new video...")
            try:
                # Load the original video to extract its audio clip
                original_clip = VideoFileClip(source_path)
                # Load the processed background-free video
                processed_clip = VideoFileClip(destination_path)
                # Set the audio of the new video to be the audio from the original video
                final_clip = processed_clip.set_audio(original_clip.audio)
                # Export the final file with the merged media
                final_clip.write_videofile(
                    destination_path2, 
                    codec="libx264", 
                    audio_codec="aac",
                    logger=None # Suppresses overly verbose command line output
                )
                # Close clips to free up system memory and file hooks
                original_clip.close()
                processed_clip.close()
                final_clip.close()
                print(f"Success! Final video saved to: {destination_path2}")
        
            except Exception as e:
                print(f"An error occurred during audio stitching: {e}")
            count = count + 1
                   
            
        
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
    ###################
    source_folder = "YouTube_video_final" 
    destination_folder = "YouTube_video_noBackground" 
    destination_folder2 = "YouTube_video_noBackground_audio" 
    remove_background(source_folder, destination_folder, destination_folder2)
    print("Video download complete")

###############################################################

    
if __name__ == '__main__':
    main()

    
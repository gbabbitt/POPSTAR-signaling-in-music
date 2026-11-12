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
import random as rnd

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Arc
import pandas as pd

import soundfile
from scipy.io import wavfile
from scipy import signal
#import noisereduce as nr
from pydub import AudioSegment
from pydub.playback import play
#from scipy.linalg import svd
from PIL import Image
import cv2  # pip install opencv-python
import shutil
from moviepy.editor import *
from PyQt5 import QtCore, QtGui, QtWidgets
import time

from moviepy.editor import VideoFileClip, clips_array

if not os.path.exists('popstar_results'):
        os.mkdir('popstar_results')

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
    if(header == "metro"):
        met = value
        print("my metronome option is",met)
    if(header == "fileExt"):
        ext = value
        print("my file extension(s) are",ext)    
 ###### variable assignments ######
inp = ""+name+""
tm = int(tm)
fof = ""+fof+""
met = ""+met+""
ext = ""+ext+""

#lyr = "no"

if(fof=="file"):
    # calculate number of faces for single file
    lst = os.listdir("%s_analysis/texts/" % inp) # your directory path
    frame_num = len(lst)  # note folder has 4 types of files
    print("number of movie frames is %s" % frame_num)
    dur = 1
    frameSec = (frame_num/dur)
    print("frames per second is %s" % frameSec)
###############################################################

def renderTextMovie():
    print("rendering movie TEXTS")
    folder = "%s_analysis/texts" % inp
    video_filename = "%s_analysis/myMovieSound_texts.mp4" % inp
    valid_images = [i for i in os.listdir(folder) if i.endswith((".jpg", ".jpeg", ".png"))]
    #print(valid_images)
    each_image_duration = 1 # 1 second
    first_image = cv2.imread(os.path.join(folder, valid_images[0]))
    h, w, _ = first_image.shape
    codec = cv2.VideoWriter_fourcc(*'mp4v')
    vid_writer = cv2.VideoWriter(video_filename, codec, 8.0, (w, h)) 
    for i in range(len(valid_images)):
        adjCNT = str(i)
        img = "text_%s.png" % adjCNT
        loaded_img = cv2.imread(os.path.join(folder, img))
        if ((loaded_img.shape[1], loaded_img.shape[0]) != (w, h)):
            loaded_img = cv2.resize(loaded_img, (w, h), interpolation=cv2.INTER_LANCZOS4)
        for _ in range(each_image_duration):
            vid_writer.write(loaded_img)
    vid_writer.release()

def renderTextMovie_batch():
    print("rendering movie TEXTS")
    lst = os.listdir(inp) # your directory path
    number_files = len(lst)
    print("number of files")
    print(number_files)
    dir_list = os.listdir(inp)
    print(dir_list)
    trk1 = 0
    for fname in dir_list:
        print(fname)
        dirname = fname[:-4]
        folder = "%s_analysis/texts/%s" % (inp,dirname)
        print(folder)
        video_filename = "%s_analysis/myMovieSound_texts_%s.mp4" % (inp,dirname)
        valid_images = [i for i in os.listdir(folder) if i.endswith((".jpg", ".jpeg", ".png"))]
        #print(valid_images)
        each_image_duration = 1 # 1 second
        first_image = cv2.imread(os.path.join(folder, valid_images[0]))
        h, w, _ = first_image.shape
        codec = cv2.VideoWriter_fourcc(*'mp4v')
        vid_writer = cv2.VideoWriter(video_filename, codec, 8.0, (w, h)) 
        for i in range(len(valid_images)):
            adjCNT = str(i)
            img = "text_%s.png" % adjCNT
            loaded_img = cv2.imread(os.path.join(folder, img))
            if ((loaded_img.shape[1], loaded_img.shape[0]) != (w, h)):
                loaded_img = cv2.resize(loaded_img, (w, h), interpolation=cv2.INTER_LANCZOS4)
            for _ in range(each_image_duration):
                vid_writer.write(loaded_img)
        vid_writer.release()




def renderTplotMovie():
    print("rendering movie TPLOT 1")
    folder = "%s_analysis/tplots1" % inp
    video_filename = "%s_analysis/myMovieSound_tplots1.mp4" % inp
    valid_images = [i for i in os.listdir(folder) if i.endswith((".jpg", ".jpeg", ".png"))]
    #print(valid_images)
    each_image_duration = 1 # 1 second
    first_image = cv2.imread(os.path.join(folder, valid_images[0]))
    h, w, _ = first_image.shape
    codec = cv2.VideoWriter_fourcc(*'mp4v')
    vid_writer = cv2.VideoWriter(video_filename, codec, 8.0, (w, h)) 
    for i in range(len(valid_images)):
        adjCNT = str(i)
        img = "tplot_%s.png" % adjCNT
        loaded_img = cv2.imread(os.path.join(folder, img))
        for _ in range(each_image_duration):
            vid_writer.write(loaded_img)
    vid_writer.release()
    
    ############################
    if(ext == ".mp4" or ext == ".mp3" or ext == ".wav" or ext == ".txt"):  
        print("rendering movie TPLOT 2")
        folder = "%s_analysis/tplots2" % inp
        video_filename = "%s_analysis/myMovieSound_tplots2.mp4" % inp
        valid_images = [i for i in os.listdir(folder) if i.endswith((".jpg", ".jpeg", ".png"))]
        #print(valid_images)
        each_image_duration = 1 # 1 second
        first_image = cv2.imread(os.path.join(folder, valid_images[0]))
        h, w, _ = first_image.shape
        codec = cv2.VideoWriter_fourcc(*'mp4v')
        vid_writer = cv2.VideoWriter(video_filename, codec, 8.0, (w, h)) 
        for i in range(len(valid_images)):
            adjCNT = str(i)
            img = "tplot_%s.png" % adjCNT
            loaded_img = cv2.imread(os.path.join(folder, img))
            for _ in range(each_image_duration):
                vid_writer.write(loaded_img)
        vid_writer.release()
    
def renderTplotMovie_batch():
    print("rendering movie 1")
    lst = os.listdir(inp) # your directory path
    number_files = len(lst)
    print("number of files")
    print(number_files)
    dir_list = os.listdir(inp)
    print(dir_list)
    trk1 = 0
    for fname in dir_list:
        print(fname)
        dirname = fname[:-4]
        folder = "%s_analysis/tplots1/%s" % (inp,dirname)
        video_filename = "%s_analysis/myMovieSound_tplots1_%s.mp4" % (inp,dirname)
        valid_images = [i for i in os.listdir(folder) if i.endswith((".jpg", ".jpeg", ".png"))]
        #print(valid_images)
        each_image_duration = 1 # 1 second
        first_image = cv2.imread(os.path.join(folder, valid_images[0]))
        h, w, _ = first_image.shape
        lstINT = os.listdir("%s_analysis/texts/%s" % (inp,dirname)) # your directory path
        frame_num = int(len(lstINT)) 
        print("number of movie frames is %s" % frame_num)
        # make video        
        codec = cv2.VideoWriter_fourcc(*'mp4v')
        vid_writer = cv2.VideoWriter(video_filename, codec, 8.0, (w, h))  # convert to constant rate of 0.125 sec
        for i in range(len(valid_images)):
            adjCNT = str(i)
            img = "tplot_%s.png" % adjCNT
            loaded_img = cv2.imread(os.path.join(folder, img))
            for _ in range(each_image_duration):
                vid_writer.write(loaded_img)

        vid_writer.release()
        trk1 = trk1 + 1
    ############################
    
    if(ext == ".mp4" or ext == ".mp3" or ext == ".wav" or ext == ".txt"):  
        print("rendering movie 2")
        lst = os.listdir(inp) # your directory path
        number_files = len(lst)
        print("number of files")
        print(number_files)
        dir_list = os.listdir(inp)
        print(dir_list)
        trk2 = 0
        for fname in dir_list:
            print(fname)
            dirname = fname[:-4]
            folder = "%s_analysis/tplots2/%s" % (inp,dirname)
            video_filename = "%s_analysis/myMovieSound_tplots2_%s.mp4" % (inp,dirname)
            valid_images = [i for i in os.listdir(folder) if i.endswith((".jpg", ".jpeg", ".png"))]
            #print(valid_images)
            each_image_duration = 1 # 1 second
            first_image = cv2.imread(os.path.join(folder, valid_images[0]))
            h, w, _ = first_image.shape
            lstINT = os.listdir("%s_analysis/texts/%s" % (inp,dirname)) # your directory path
            frame_num = int(len(lstINT))
            print("number of movie frames is %s" % frame_num)
            # make video        
            codec = cv2.VideoWriter_fourcc(*'mp4v')
            vid_writer = cv2.VideoWriter(video_filename, codec, 8.0, (w, h))  # convert to constant rate of 0.125 sec
            for i in range(len(valid_images)):
                adjCNT = str(i)
                img = "tplot_%s.png" % adjCNT
                loaded_img = cv2.imread(os.path.join(folder, img))
                for _ in range(each_image_duration):
                    vid_writer.write(loaded_img)

            vid_writer.release()
            trk2 = trk2 + 1
                
    

def combine_side_by_side():
    # Load the two video clips
    clip0 = VideoFileClip("%s_analysis/myMovieSound_tplots1.mp4" % inp)
    clip1 = VideoFileClip("%s_analysis/myMovieSound_texts.mp4" % inp)
    clip2 = VideoFileClip("%s_analysis/myMovieSound_tplots2.mp4" % (inp))
    # Ensure both clips have the same height for side-by-side alignment
    clip1 = clip1.resize(height=clip0.h)
    clip2 = clip2.resize(height=clip0.h)

    # Concatenate the clips side by side
    final_clip = clips_array([[clip0, clip1, clip2]])
    
    # Write the combined video to a new file
    final_clip.write_videofile("%s_analysis/myMovieSound_combined.mp4" % inp)

def combine_side_by_side_batch():
    lst = os.listdir(inp) # your directory path
    number_files = len(lst)
    print("number of files")
    print(number_files)
    dir_list = os.listdir(inp)
    print(dir_list)
    for fname in dir_list:
        print(fname)
        dirname = fname[:-4]
        # Load the two video clips
        clip0 = VideoFileClip("%s_analysis/myMovieSound_tplots1_%s.mp4" % (inp,dirname))
        clip1 = VideoFileClip("%s_analysis/myMovieSound_texts_%s.mp4" % (inp,dirname))
        clip2 = VideoFileClip("%s_analysis/myMovieSound_tplots2_%s.mp4" % (inp,dirname))
        # Ensure both clips have the same height for side-by-side alignment
        clip1 = clip1.resize(height=clip0.h)
        clip2 = clip2.resize(height=clip0.h)

        # Concatenate the clips side by side
        final_clip = clips_array([[clip0, clip1, clip2]])
        
        # Write the combined video to a new file
        final_clip.write_videofile("%s_analysis/myMovieSound_combined_%s.mp4" % (inp,dirname))



def copyMovie():
    print("loading movie")
    shutil.copy2("%s_analysis/myMovieSound_texts.mp4" % inp, "popstar_results/myMovie_faces_%s.mp4" % inp)
    shutil.copy2("%s_analysis/myMovieSound_tplots1.mp4" % inp, "popstar_results/myMovie_tplots1_%s.mp4" % inp)
    if(ext == ".mp4" or ext == ".mp3" or ext == ".wav" or ext == ".txt"): 
        shutil.copy2("%s_analysis/myMovieSound_tplots2.mp4" % inp, "popstar_results/myMovie_tplots2_%s.mp4" % inp)
    shutil.copy2("%s_analysis/myMovieSound_combined.mp4" % inp, "popstar_results/myMovie_combo_%s.mp4" % inp)

def copyMovie_batch():
    print("loading movie")
    lst = os.listdir(inp) # your directory path
    number_files = len(lst)
    print("number of files")
    print(number_files)
    dir_list = os.listdir(inp)
    print(dir_list)
    for fname in dir_list:
        print(fname)
        dirname = fname[:-4]
        shutil.copy2("%s_analysis/myMovieSound_texts_%s.mp4" % (inp,dirname), "popstar_results/myMovie_texts_%s.mp4" % (dirname))
        shutil.copy2("%s_analysis/myMovieSound_tplots1_%s.mp4" % (inp,dirname), "popstar_results/myMovie_tplots1_%s.mp4" % (dirname))
        if(ext == ".mp4" or ext == ".mp3" or ext == ".wav" or ext == ".txt"): 
            shutil.copy2("%s_analysis/myMovieSound_tplots2_%s.mp4" % (inp,dirname), "popstar_results/myMovie_tplots2_%s.mp4" % (dirname))
        shutil.copy2("%s_analysis/myMovieSound_combined_%s.mp4" % (inp,dirname), "popstar_results/myMovie_combo_%s.mp4" % (dirname))

def distances():
    print("calc distances")
    readPath = "%s_analysis/ternary_norm.txt" % (inp)
    writePath = "%s_analysis/distances.txt" % (inp)
    writePath2 = "%s_analysis/distances_order1.txt" % (inp)
    writePath3 = "%s_analysis/distances_order0.txt" % (inp)
    txt_out = open(writePath, "w")
    txt_out2 = open(writePath2, "w")
    txt_out3 = open(writePath3, "w")
    txt_out.write("order\tdistance\n")
    txt_out2.write("distance\n")
    txt_out3.write("distance\n")
    df = pd.read_csv(readPath)
    print(df)
    #df_array = np.array(df)
    df_bootstrap = df.sample(n = len(df), axis='index', replace=True)
    distances = []
    for i in range(len(df)-1):
        x2 = df.iloc[i+1,0]
        x1 = df.iloc[i,0]
        y2 = df.iloc[i+1,1]
        y1 = df.iloc[i,1]
        z2 = df.iloc[i+1,2]
        z1 = df.iloc[i,2]
        dist = np.sqrt((x2-x1)**2 + (y2-y1)**2 +(z2-z1)**2)
        txt_out.write("first_order\t%s\n" % str(dist))
        txt_out2.write("%s\n" % str(dist))
        distances.append(dist)
    #print(distances)
    for i in range(len(df_bootstrap)-1):
        x2 = df_bootstrap.iloc[i+1,0]
        x1 = df_bootstrap.iloc[i,0]
        y2 = df_bootstrap.iloc[i+1,1]
        y1 = df_bootstrap.iloc[i,1]
        z2 = df_bootstrap.iloc[i+1,2]
        z1 = df_bootstrap.iloc[i,2]
        dist = np.sqrt((x2-x1)**2 + (y2-y1)**2 +(z2-z1)**2)
        txt_out.write("zero_order\t%s\n" % str(dist))
        txt_out3.write("%s\n" % str(dist))
        distances.append(dist)
    #print(distances)
    txt_out.close
    txt_out2.close
    txt_out3.close
    
def distances_batch():
    print("calc distances")
    lst = os.listdir("%s_analysis/texts/" % (inp)) # your directory path
    #lst = os.listdir(inp) # your directory path
    number_files = len(lst)
    print("number of files")
    print(number_files)
    dir_list = os.listdir("%s_analysis/texts/" % (inp))
    print(dir_list)
    for fname in dir_list:
        print(fname)
        dirname = fname
        readPath = "%s_analysis/ternary_norm_%s.txt" % (inp,dirname)
        writePath = "%s_analysis/distances_%s.txt" % (inp,dirname)
        writePath2 = "%s_analysis/distances_order1_%s.txt" % (inp,dirname)
        writePath3 = "%s_analysis/distances_order0_%s.txt" % (inp,dirname)
        txt_out = open(writePath, "w")
        txt_out2 = open(writePath2, "w")
        txt_out3 = open(writePath3, "w")
        txt_out.write("order\tdistance\n")
        txt_out2.write("distance\n")
        txt_out3.write("distance\n")
        df = pd.read_csv(readPath)
        print(df)
         #df_array = np.array(df)
        df_bootstrap = df.sample(n = len(df), axis='index', replace=True)
        distances = []
        for i in range(len(df)-1):
            x2 = df.iloc[i+1,0]
            x1 = df.iloc[i,0]
            y2 = df.iloc[i+1,1]
            y1 = df.iloc[i,1]
            z2 = df.iloc[i+1,2]
            z1 = df.iloc[i,2]
            dist = np.sqrt((x2-x1)**2 + (y2-y1)**2 +(z2-z1)**2)
            txt_out.write("first_order\t%s\n" % str(dist))
            txt_out2.write("%s\n" % str(dist))
            distances.append(dist)
        #print(distances)
        for i in range(len(df_bootstrap)-1):
            x2 = df_bootstrap.iloc[i+1,0]
            x1 = df_bootstrap.iloc[i,0]
            y2 = df_bootstrap.iloc[i+1,1]
            y1 = df_bootstrap.iloc[i,1]
            z2 = df_bootstrap.iloc[i+1,2]
            z1 = df_bootstrap.iloc[i,2]
            dist = np.sqrt((x2-x1)**2 + (y2-y1)**2 +(z2-z1)**2)
            txt_out.write("zero_order\t%s\n" % str(dist))
            txt_out3.write("%s\n" % str(dist))
            distances.append(dist)
        #print(distances)
        txt_out.close
        txt_out2.close
        txt_out3.close
        

 
 
 
        
##############################################################    
def main():
    if(fof == "file"):
        renderTextMovie()
        renderTplotMovie()
        combine_side_by_side()
        copyMovie()
        distances()
            
    if(fof == "folder"):
        renderTextMovie_batch()
        renderTplotMovie_batch()
        combine_side_by_side_batch()
        copyMovie_batch()
        distances_batch()
           
###############################################################
if __name__ == '__main__':
    main()
    
    
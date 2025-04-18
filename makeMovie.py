#!/usr/bin/env python

#############################################################################
######   POPSTAR software for detecting fitness signaling in music
######   produced by Dr. Gregory A. Babbitt
######   and students at the Rochester Instituteof Technology in 2022.
######   Offered freely without guarantee.  License under GPL v3.0
#############################################################################


import getopt, sys # Allows for command line arguments
import os
import shutil
import random as rnd

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Arc

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

# video window
from PyQt5.QtCore import QDir, Qt, QUrl
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtWidgets import (QApplication, QFileDialog, QHBoxLayout, QLabel,
        QPushButton, QSizePolicy, QSlider, QStyle, QVBoxLayout, QWidget)
from PyQt5.QtWidgets import QMainWindow,QWidget, QPushButton, QAction
from PyQt5.QtGui import QIcon


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

# calculate number of faces for single file
lst = os.listdir("%s_analysis/intervals/" % inp) # your directory path
frame_num = int(len(lst)/4)  # note folder has 4 types of files
print("number of movie frames is %s" % frame_num)
# calculate number of faces for folder

def renderFaceMovie():
    print("rendering movie")
    folder = "%s_analysis/faces" % inp
    video_filename = "%s_analysis/myMovie_faces.mp4" % inp
    valid_images = [i for i in os.listdir(folder) if i.endswith((".jpg", ".jpeg", ".png"))]
    #print(valid_images)
    each_image_duration = 1 # 1 second
    first_image = cv2.imread(os.path.join(folder, valid_images[0]))
    h, w, _ = first_image.shape

    codec = cv2.VideoWriter_fourcc(*'mp4v')
    vid_writer = cv2.VideoWriter(video_filename, codec, 4.0, (w, h))  # convert to constant rate of 0.5 sec

    for img in valid_images:
        loaded_img = cv2.imread(os.path.join(folder, img))
        for _ in range(each_image_duration):
            vid_writer.write(loaded_img)

    vid_writer.release()
    
def faceMovie_audio_video():
    # map MMD in chimerax
    print("combining audio and video for movie for %s" % inp)
    audio_file = "%s_analysis/trimmed_%s.wav" % (inp,inp)
    video_file = "%s_analysis/myMovie_faces.mp4" % inp
    wave_file = AudioSegment.from_file('%s_analysis/trimmed_%s.wav' % (inp,inp))
    #wave_file_trim = wave_file[0000:8000] # 8 second fit to movie file             
    #wave_file_trim.export('proteinInteraction_movie_%s/mySound_trim.wav' % PDB_id_reference, format="wav")
    #audio_file = "proteinInteraction_movie_%s/mySound_trim.wav" % PDB_id_reference
    # load the video
    video_clip = VideoFileClip(video_file)
    # load the audio
    audio_clip = AudioFileClip(audio_file)
    #start = 0
    # if end is not set, use video clip's end
    #end = video_clip.end
    
    # setting the start & end of the audio clip to `start` and `end` paramters
    #audio_clip = audio_clip.subclip(start, end)
    # add the final audio to the video
    final_clip = video_clip.set_audio(audio_clip)
    # save the final clip
    final_clip.write_videofile("%s_analysis/myMovieSound_faces.mp4" % inp)
    
def copyFaceMovie():
    print("loading movie")
    shutil.copy2("%s_analysis/myMovieSound_faces.mp4" % inp, "myMovie_faces.mp4")
    
    
def main():
    renderFaceMovie()
    faceMovie_audio_video()
    copyFaceMovie()
    

###############################################################
if __name__ == '__main__':
    main()
    
    
#!/usr/bin/env python

#############################################################################
######   POPSTAR software for detecting fitness signaling in music
######   produced by Dr. Gregory A. Babbitt
######   and students at the Rochester Instituteof Technology in 2025.
######   Offered freely without guarantee.  License under GPL v3.0
#############################################################################


import getopt, sys # Allows for command line arguments
import os
import gc
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import math
import random as rnd
import re
import cv2
import time
from pydub import AudioSegment
import soundfile
import librosa 
# for ggplot
from plotnine import *
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
#inp = "test"
#tm = 20 # interval length in seconds

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
    if(header == "metro"):
        met = value
        print("my metronome option is",met)
    if(header == "spch"):
        spchOpt = value
        print("speech-normalize",spchOpt)
    if(header == "self"):
        selfOpt = value
        print("self-normalize",selfOpt)
    if(header == "musi"):
        musiOpt = value
        print("music-normalize",musiOpt)
    if(header == "fileExt"):
        ext = value
        print("my file extension(s) are",ext)   
infile.close()
 ###### variable assignments ######
inp = ""+name+""
tm = int(tm)
met = ""+met+""
spchOpt = ""+spchOpt+""
selfOpt = ""+selfOpt+""
musiOpt = ""+musiOpt+""
ext = ""+ext+""

'''
if os.path.exists('%s_analysis' % inp):
    print("folder already exists...was this run already done?")
    exit()
'''
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
input0 = "%s.wav" % inp
input1 = "%s.mp4" % inp
input2 = "%s.png" % inp
input3 = "%s.dat" % inp
input4 = "%s.txt" % inp
input5 = "%s.jpg" % inp

if os.path.isfile(input1):
    print("user input is a .mp4 file")
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
def GlobOptContrast():
    print("\ncalculating global optical contrast on %s\n" % input1)
    cap = cv2.VideoCapture(input1)
    fps = cap.get(cv2.CAP_PROP_FPS) # frames per second
    print(f"Video FPS: {fps}")
    fpw= tm*fps # frames per window
    print(f"Video FPW: {fpw}")
    tfs = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # total frames
    print(f"Total Frames: {tfs}")
    if not cap.isOpened():
        print("Error: Could not open video.")
        return []

    std_contrast_values = []
    michelson_contrast_values = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        #print(frame)
        # Convert the frame to grayscale for intensity calculation
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 1. Standard Deviation Method (Highly reliable for global contrast)
        std_contrast = float(np.std(gray_frame))
        # 2. Michelson Contrast Method (Range between darkest and brightest spots)
        min_val, max_val, _, _ = cv2.minMaxLoc(gray_frame)
        # Avoid division by zero if the frame is completely black
        denominator = max_val + min_val
        michelson_contrast = (max_val - min_val) / denominator if denominator > 0 else 0
        # Append the mean contrast for frame
        std_contrast_values.append(std_contrast)
        michelson_contrast_values.append(michelson_contrast)

    # average over whole .mp4
    #print(std_contrast_values)
    avg_std_contrast = np.mean(std_contrast_values)
    print("avg_std_contrast")
    print(avg_std_contrast)
    #print(michelson_contrast_values)
    #total_michelson_contrast = np.mean(michelson_contrast_values)
    #print("total_michelson_contrast") 
    #print(total_michelson_contrast) 
    ################################################
    ###  collect optical contrast over windows
    ################################################
    print("segmenting %s" % input1)
    print("length of file (seconds)")
    #tm = 20 # interval length in seconds
    file_path = "%s_analysis/trimmed_%s" % (inp,input0)
    song = AudioSegment.from_file("%s_analysis/trimmed_%s" % (inp,input0), format="wav") 
    print(song.duration_seconds)
    dur = song.duration_seconds
    # Estimate the tempo (BPM)
    y, sr = librosa.load(file_path)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo = tempo[0]
    total_beats = (dur/60*tempo)
    beat_int = dur/total_beats
    print("tempo = %s bpm" % tempo)
    print("beat interval = %s sec" % beat_int)
    print("total beats = %s" % total_beats)
    #f = open("./popstar.ctl", "a")
    #f.write("duration,%s,#song duration (seconds)\n" % dur)
    #f.write("tempo,%s,#tempo (bpm)\n" % tempo)
    #f.write("beatInt,%s,#beat interval (seconds)\n" % beat_int)
    #f.write("ttlBeats,%s,#total number beats in song\n" % total_beats)
    #f.close()
    #print(myStop)
    if(met == "no"):
        ints = int(dur*8)-tm  # analyze in 1/8 second fixed sliding window
    #ints = int(dur*4*beat_int)-tm  # analyze fixed sliding window attempting matching beat intervals
    if(met == "yes"):   
        ints = int(total_beats)
    # start and end time 
    global opt_contrast_windows
    opt_contrast_windows = []
    for i in range(ints): 
        if(met == "no"):
            start = i*125  # note 250 = 0.125 second fixed window
            end = i*125+tm*1000
            start_video = start
            end_video = end/1000
        if(met == "yes"):
            start = i*beat_int*1000  # attempt to match beat intervals
            end = i*beat_int*1000+tm*1000
            start_video = start/1000
            end_video = end/1000
        #print("audio start: %s end: %s" % (start,end))
        start_frames = int(start_video*fps)
        end_frames = int(end_video*fps)
        if(end_frames >= tfs):
            end_frames = tfs
        print("video start: %s end: %s (secs)" % (start_video,end_video))
        print("frames start: %s end: %s (count)" % (start_frames,end_frames))
        window = std_contrast_values[start_frames:end_frames]
        opt_contrast_window = np.mean(window)
        print("opt_contrast: %s" % opt_contrast_window)
        opt_contrast_windows.append(float(opt_contrast_window) )
    #print(opt_contrast_windows)
    #write to file
    print("\nwriting contrast values\n")
    f = open("%s_analysis/CONTRASTvalues_%s.txt" % (inp,inp), "w")
    f.write("optical_contrast\n")
    for i in range(len(opt_contrast_windows)):
        print("%s\t%s" % (i,opt_contrast_windows[i]))
        f.write("%s\n" % (opt_contrast_windows[i]))
    f.close()
    cap.release()

def GlobOptContrast_batch():
    lst = os.listdir(inp) # your directory path
    number_files = len(lst)
    print("number of files")
    print(number_files)
    dir_list = os.listdir(inp)
    print(dir_list)
    for i in range(number_files):    
        # Open an mp3 file 
        filename = dir_list[i]
        print("\ncalculating global optical contrast on %s\n" % filename)
        cap = cv2.VideoCapture("%s/%s" % (inp,filename))
        fps = cap.get(cv2.CAP_PROP_FPS) # frames per second
        print(f"Video FPS: {fps}")
        fpw= tm*fps # frames per window
        print(f"Video FPW: {fpw}")
        tfs = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # total frames
        print(f"Total Frames: {tfs}")
        if not cap.isOpened():
            print("Error: Could not open video.")
            return []

        std_contrast_values = []
        michelson_contrast_values = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            #print(frame)
            # Convert the frame to grayscale for intensity calculation
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # 1. Standard Deviation Method (Highly reliable for global contrast)
            std_contrast = float(np.std(gray_frame))
            # 2. Michelson Contrast Method (Range between darkest and brightest spots)
            min_val, max_val, _, _ = cv2.minMaxLoc(gray_frame)
            # Avoid division by zero if the frame is completely black
            denominator = max_val + min_val
            michelson_contrast = (max_val - min_val) / denominator if denominator > 0 else 0
             # Append the mean contrast for frame
            std_contrast_values.append(std_contrast)
            michelson_contrast_values.append(michelson_contrast)

        # average over whole .mp4
        #print(std_contrast_values)
        avg_std_contrast = np.mean(std_contrast_values)
        print("avg_std_contrast")
        print(avg_std_contrast)
        #print(michelson_contrast_values)
        #total_michelson_contrast = np.mean(michelson_contrast_values)
        #print("total_michelson_contrast") 
        #print(total_michelson_contrast) 
        ################################################
        ###  collect optical contrast over windows
        ################################################
        print("segmenting %s" % filename)
        print("length of file (seconds)")
        #tm = 20 # interval length in seconds
        file_path = "%s_analysis/%s.wav" % (inp,filename[:-4])
        song = AudioSegment.from_file("%s_analysis/%s.wav" % (inp,filename[:-4]), format="wav") 
        print(song.duration_seconds)
        dur = song.duration_seconds
        # Estimate the tempo (BPM)
        y, sr = librosa.load(file_path)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo = tempo[0]
        total_beats = (dur/60*tempo)
        beat_int = dur/total_beats
        print("tempo = %s bpm" % tempo)
        print("beat interval = %s sec" % beat_int)
        print("total beats = %s" % total_beats)
        #f = open("./popstar.ctl", "a")
        #f.write("duration,%s,#song duration (seconds)\n" % dur)
        #f.write("tempo,%s,#tempo (bpm)\n" % tempo)
        #f.write("beatInt,%s,#beat interval (seconds)\n" % beat_int)
        #f.write("ttlBeats,%s,#total number beats in song\n" % total_beats)
        #f.close()
        #print(myStop)
        if(met == "no"):
            ints = int(dur*8)-tm  # analyze in 1/8 second fixed sliding window
        #ints = int(dur*4*beat_int)-tm  # analyze fixed sliding window attempting matching beat intervals
        if(met == "yes"):   
            ints = int(total_beats)
        # start and end time 
        global opt_contrast_windows
        opt_contrast_windows = []
        for i in range(ints): 
            if(met == "no"):
                start = i*125  # note 250 = 0.125 second fixed window
                end = i*125+tm*1000
                start_video = start
                end_video = end/1000
            if(met == "yes"):
                start = i*beat_int*1000  # attempt to match beat intervals
                end = i*beat_int*1000+tm*1000
                start_video = start/1000
                end_video = end/1000
            #print("audio start: %s end: %s" % (start,end))
            start_frames = int(start_video*fps)
            end_frames = int(end_video*fps)
            if(end_frames >= tfs):
                end_frames = tfs
            print("video start: %s end: %s (secs)" % (start_video,end_video))
            print("frames start: %s end: %s (count)" % (start_frames,end_frames))
            window = std_contrast_values[start_frames:end_frames]
            opt_contrast_window = np.mean(window)
            print("opt_contrast: %s" % opt_contrast_window)
            opt_contrast_windows.append(float(opt_contrast_window) )
        #print(opt_contrast_windows)
        #write to file
        print("\nwriting contrast values\n")
        f = open("%s_analysis/CONTRASTvalues_%s.txt" % (inp,filename[:-4]), "w")
        f.write("optical_contrast\n")
        for i in range(len(opt_contrast_windows)):
            print("%s\t%s" % (i,opt_contrast_windows[i]))
            f.write("%s\n" % (opt_contrast_windows[i]))
        f.close()
        cap.release()
        cv2.destroyAllWindows()

def OptFlow2():
    print("\ncalculating dense optical flow (Farneback) on %s\n" % input1)
    cap0 = cv2.VideoCapture(input1)
    fps = cap0.get(cv2.CAP_PROP_FPS) # frames per second
    print(f"Video FPS: {fps}")
    fpw= tm*fps # frames per window
    print(f"Video FPW: {fpw}")
    fpw= tm*fps # frames per window
    print(f"Video FPW: {fpw}")
    tfs = int(cap0.get(cv2.CAP_PROP_FRAME_COUNT))  # total frames
    print(f"Total Frames: {tfs}")
    if not cap0.isOpened():
        print("Error: Could not open video.")
        return []
    cap0.release()
    cv2.destroyAllWindows()
    ################################################
    ###  collect optical flow over windows
    ################################################
    print("segmenting %s" % input1)
    print("length of file (seconds)")
    #tm = 20 # interval length in seconds
    file_path = "%s_analysis/trimmed_%s" % (inp,input0)
    song = AudioSegment.from_file("%s_analysis/trimmed_%s" % (inp,input0), format="wav") 
    print(song.duration_seconds)
    dur = song.duration_seconds
    # Estimate the tempo (BPM)
    y, sr = librosa.load(file_path)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo = tempo[0]
    total_beats = (dur/60*tempo)
    beat_int = dur/total_beats
    print("tempo = %s bpm" % tempo)
    print("beat interval = %s sec" % beat_int)
    print("total beats = %s" % total_beats)
    #f = open("./popstar.ctl", "a")
    #f.write("duration,%s,#song duration (seconds)\n" % dur)
    #f.write("tempo,%s,#tempo (bpm)\n" % tempo)
    #f.write("beatInt,%s,#beat interval (seconds)\n" % beat_int)
    #f.write("ttlBeats,%s,#total number beats in song\n" % total_beats)
    #f.close()
    #print(myStop)
    if(met == "no"):
        ints = int(dur*8)-tm  # analyze in 1/8 second fixed sliding window
    #ints = int(dur*4*beat_int)-tm  # analyze fixed sliding window attempting matching beat intervals
    if(met == "yes"):   
        ints = int(total_beats)
    # start and end time 
    opt_flow_windows = []
    dlt_flow_windows = []
    for i in range(ints): 
        print("\nanalyzing time slice %s/%s\n" % (i,ints))
        time.sleep(1)
        # delete time-slice.mp4 to prevent ffmpeg overwrite prompts
        removal_path = "time_slice.mp4"
        if os.path.exists(removal_path):
            os.remove(removal_path)
        if(met == "no"):
            start = i*125  # note 250 = 0.125 second fixed window
            end = i*125+tm*1000
            start_video = start
            end_video = end/1000
        if(met == "yes"):
            start = i*beat_int*1000  # attempt to match beat intervals
            end = i*beat_int*1000+tm*1000
            start_video = start/1000
            end_video = end/1000
        #print("audio start: %s end: %s" % (start,end))
        start_frames = int(start_video*fps)
        end_frames = int(end_video*fps)
        if(end_frames >= tfs):
            end_frames = tfs
        print("video start: %s end: %s (secs)" % (start_video,end_video))
        print("frames start: %s end: %s (count)" % (start_frames,end_frames))
        hours, remainder = divmod(start_video, 3600)
        minutes, secs = divmod(remainder, 60)
        hours = str(int(hours)).zfill(2)
        minutes = str(int(minutes)).zfill(2)
        secs = str(int(secs)).zfill(2)
        start_time_str = f"{hours}:{minutes}:{secs}"
        start_time_str = str(start_video)
        #print(start_time_str)
        hours, remainder = divmod(end_video, 3600)
        minutes, secs = divmod(remainder, 60)
        hours = str(int(hours)).zfill(2)
        minutes = str(int(minutes)).zfill(2)
        secs = str(int(secs)).zfill(2)
        end_time_str = f"{hours}:{minutes}:{secs}"
        end_time_str = str(end_video)
        #print(end_time_str)
        # extract video for sliding window
        cmd = "ffmpeg -ss %s -to %s -i %s -c copy time_slice.mp4" % (start_time_str,end_time_str,input1)
        os.system(cmd)
        cap = cv2.VideoCapture("time_slice.mp4")
        if not cap.isOpened():
            print("Error: Could not open video.")
            return []
        opt_flow_values = []
        dlt_flow_values = []
        # first frame
        ret, frame = cap.read()
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            #print(frame)
            # Convert the frame to grayscale for intensity calculation
            gray_frame_next = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(gray_frame, gray_frame_next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1],angleInDegrees=True)
            opt_flow_values.append(magnitude)
            dlt_flow_values.append(angle)
            gray_frame = gray_frame_next
                
        avg_flow_magnitude = np.mean(opt_flow_values)
        std_flow_angle = np.std(dlt_flow_values)
        print("opt_flow: %s" % avg_flow_magnitude)
        print("dlt_flow: %s" % std_flow_angle)
        opt_flow_windows.append(float(avg_flow_magnitude))
        dlt_flow_windows.append(float(std_flow_angle))
        cap.release()
        cv2.destroyAllWindows()
        gc.collect()
        removal_path = "time_slice.mp4"
        if os.path.exists(removal_path):
            os.remove(removal_path)
    #print(opt_flow_windows)
    #print(dlt_flow_windows)
    # write to file
    print("\nwriting flow values\n")
    f = open("%s_analysis/FLOWvalues_%s.txt" % (inp,inp), "w")
    f.write("optical_flow\tdelta_flow\n")
    for i in range(len(opt_flow_windows)):
        print("%s\t%s\t%s" % (i,opt_flow_windows[i], dlt_flow_windows[i]))
        f.write("%s\t%s\n" % (opt_flow_windows[i], dlt_flow_windows[i]))
    f.close()
      
    print("\ncompleted dense optical flow (Farneback) on %s\n" % input1)

    
def OptFlow():
    print("\ncalculating dense optical flow (Farneback) on %s\n" % input1)
    cap0 = cv2.VideoCapture(input1)
    fps = cap0.get(cv2.CAP_PROP_FPS) # frames per second
    print(f"Video FPS: {fps}")
    fpw= tm*fps # frames per window
    print(f"Video FPW: {fpw}")
    fpw= tm*fps # frames per window
    print(f"Video FPW: {fpw}")
    tfs = int(cap0.get(cv2.CAP_PROP_FRAME_COUNT))  # total frames
    print(f"Total Frames: {tfs}")
    if not cap0.isOpened():
        print("Error: Could not open video.")
        return []
    opt_flow_values = []
    dlt_flow_values = []
    # first frame
    ret, frame = cap0.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    while True:
        cap = cv2.VideoCapture(input1)
        ret, frame = cap.read()
        if not ret:
            break
        #print(frame)
        # Convert the frame to grayscale for intensity calculation
        gray_frame_next = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(gray_frame, gray_frame_next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1],angleInDegrees=True)
        opt_flow_values.append(magnitude)
        dlt_flow_values.append(angle)
        gray_frame = gray_frame_next
        cap.release()
        gc.collect()
    
    avg_flow_magnitude = np.mean(opt_flow_values)
    #diff_flow_angles = np.diff(dlt_flow_values, prepend = 0)
    std_flow_angle = np.std(dlt_flow_values)
    print("avg_flow_magnitude")
    print(avg_flow_magnitude)
    print("std_flow_angle")
    print(std_flow_angle)
    
    ################################################
    ###  collect optical flow over windows
    ################################################
    print("segmenting %s" % input1)
    print("length of file (seconds)")
    #tm = 20 # interval length in seconds
    file_path = "%s_analysis/trimmed_%s" % (inp,input0)
    song = AudioSegment.from_file("%s_analysis/trimmed_%s" % (inp,input0), format="wav") 
    print(song.duration_seconds)
    dur = song.duration_seconds
    # Estimate the tempo (BPM)
    y, sr = librosa.load(file_path)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo = tempo[0]
    total_beats = (dur/60*tempo)
    beat_int = dur/total_beats
    print("tempo = %s bpm" % tempo)
    print("beat interval = %s sec" % beat_int)
    print("total beats = %s" % total_beats)
    #f = open("./popstar.ctl", "a")
    #f.write("duration,%s,#song duration (seconds)\n" % dur)
    #f.write("tempo,%s,#tempo (bpm)\n" % tempo)
    #f.write("beatInt,%s,#beat interval (seconds)\n" % beat_int)
    #f.write("ttlBeats,%s,#total number beats in song\n" % total_beats)
    #f.close()
    #print(myStop)
    if(met == "no"):
        ints = int(dur*8)-tm  # analyze in 1/8 second fixed sliding window
    #ints = int(dur*4*beat_int)-tm  # analyze fixed sliding window attempting matching beat intervals
    if(met == "yes"):   
        ints = int(total_beats)
    # start and end time 
    opt_flow_windows = []
    dlt_flow_windows = []
    for i in range(ints): 
        if(met == "no"):
            start = i*125  # note 250 = 0.125 second fixed window
            end = i*125+tm*1000
            start_video = start
            end_video = end/1000
        if(met == "yes"):
            start = i*beat_int*1000  # attempt to match beat intervals
            end = i*beat_int*1000+tm*1000
            start_video = start/1000
            end_video = end/1000
        #print("audio start: %s end: %s" % (start,end))
        start_frames = int(start_video*fps)
        end_frames = int(end_video*fps)
        if(end_frames >= tfs):
            end_frames = tfs
        print("video start: %s end: %s (secs)" % (start_video,end_video))
        print("frames start: %s end: %s (count)" % (start_frames,end_frames))
        window1 = opt_flow_values[start_frames:end_frames]
        window2 = dlt_flow_values[start_frames:end_frames]
        opt_flow_window = np.mean(window1)
        dlt_flow_window = np.std(window2)
        print("opt_flow: %s" % opt_flow_window)
        print("dlt_flow: %s" % dlt_flow_window)
        opt_flow_windows.append(float(opt_flow_window))
        dlt_flow_windows.append(float(dlt_flow_window))
        gc.collect()
    #print(opt_flow_windows)
    #print(dlt_flow_windows)
    # write to file
    print("\nwriting flow values\n")
    f = open("%s_analysis/FLOWvalues_%s.txt" % (inp,inp), "w")
    f.write("optical_flow\tdelta_flow\n")
    for i in range(len(opt_flow_windows)):
        print("%s\t%s\t%s" % (i,opt_flow_windows[i], dlt_flow_windows[i]))
        f.write("%s\t%s\n" % (opt_flow_windows[i], dlt_flow_windows[i]))
    f.close()
    cap0.release()
    cv2.destroyAllWindows()

def OptFlow_batch():
    lst = os.listdir(inp) # your directory path
    number_files = len(lst)
    print("number of files")
    print(number_files)
    dir_list = os.listdir(inp)
    print(dir_list)
    for i in range(number_files):    
        # Open an mp3 file 
        filename = dir_list[i] 
        print("\ncalculating dense optical flow (Farneback) on %s\n" % filename)
        cap0 = cv2.VideoCapture("%s/%s" % (inp,filename))
        fps = cap0.get(cv2.CAP_PROP_FPS) # frames per second
        print(f"Video FPS: {fps}")
        fpw= tm*fps # frames per window
        print(f"Video FPW: {fpw}")
        fpw= tm*fps # frames per window
        print(f"Video FPW: {fpw}")
        tfs = int(cap0.get(cv2.CAP_PROP_FRAME_COUNT))  # total frames
        print(f"Total Frames: {tfs}")
        if not cap0.isOpened():
            print("Error: Could not open video.")
            return []
        cap0.release()
        cv2.destroyAllWindows()
        ################################################
        ###  collect optical flow over windows
        ################################################
        print("segmenting %s" % filename)
        print("length of file (seconds)")
        #tm = 20 # interval length in seconds
        file_path = "%s_analysis/%s.wav" % (inp,filename[:-4])
        song = AudioSegment.from_file("%s_analysis/%s.wav" % (inp,filename[:-4]), format="wav") 
        print(song.duration_seconds)
        dur = song.duration_seconds
        # Estimate the tempo (BPM)
        y, sr = librosa.load(file_path)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo = tempo[0]
        total_beats = (dur/60*tempo)
        beat_int = dur/total_beats
        print("tempo = %s bpm" % tempo)
        print("beat interval = %s sec" % beat_int)
        print("total beats = %s" % total_beats)
        #f = open("./popstar.ctl", "a")
        #f.write("duration,%s,#song duration (seconds)\n" % dur)
        #f.write("tempo,%s,#tempo (bpm)\n" % tempo)
        #f.write("beatInt,%s,#beat interval (seconds)\n" % beat_int)
        #f.write("ttlBeats,%s,#total number beats in song\n" % total_beats)
        #f.close()
        #print(myStop)
        if(met == "no"):
            ints = int(dur*8)-tm  # analyze in 1/8 second fixed sliding window
        #ints = int(dur*4*beat_int)-tm  # analyze fixed sliding window attempting matching beat intervals
        if(met == "yes"):   
            ints = int(total_beats)
        # start and end time 
        opt_flow_windows = []
        dlt_flow_windows = []
        for i in range(ints): 
            print("\nanalyzing time slice %s/%s\n" % (i,ints))
            time.sleep(1)
            # delete time-slice.mp4 to prevent ffmpeg overwrite prompts
            removal_path = "time_slice.mp4"
            if os.path.exists(removal_path):
                os.remove(removal_path)
            if(met == "no"):
                start = i*125  # note 250 = 0.125 second fixed window
                end = i*125+tm*1000
                start_video = start
                end_video = end/1000
            if(met == "yes"):
                start = i*beat_int*1000  # attempt to match beat intervals
                end = i*beat_int*1000+tm*1000
                start_video = start/1000
                end_video = end/1000
            #print("audio start: %s end: %s" % (start,end))
            start_frames = int(start_video*fps)
            end_frames = int(end_video*fps)
            if(end_frames >= tfs):
                end_frames = tfs
            print("video start: %s end: %s (secs)" % (start_video,end_video))
            print("frames start: %s end: %s (count)" % (start_frames,end_frames))
            hours, remainder = divmod(start_video, 3600)
            minutes, secs = divmod(remainder, 60)
            hours = str(int(hours)).zfill(2)
            minutes = str(int(minutes)).zfill(2)
            secs = str(int(secs)).zfill(2)
            start_time_str = f"{hours}:{minutes}:{secs}"
            start_time_str = str(start_video)
            #print(start_time_str)
            hours, remainder = divmod(end_video, 3600)
            minutes, secs = divmod(remainder, 60)
            hours = str(int(hours)).zfill(2)
            minutes = str(int(minutes)).zfill(2)
            secs = str(int(secs)).zfill(2)
            end_time_str = f"{hours}:{minutes}:{secs}"
            end_time_str = str(end_video)
            #print(end_time_str)
            # extract video for sliding window
            cmd = "ffmpeg -ss %s -to %s -i %s/%s -c copy time_slice.mp4" % (start_time_str,end_time_str,inp,filename)
            os.system(cmd)
            cap = cv2.VideoCapture("time_slice.mp4")
            if not cap.isOpened():
                print("Error: Could not open video.")
                return []
            opt_flow_values = []
            dlt_flow_values = []
            # first frame
            ret, frame = cap.read()
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                #print(frame)
                # Convert the frame to grayscale for intensity calculation
                gray_frame_next = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                flow = cv2.calcOpticalFlowFarneback(gray_frame, gray_frame_next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1],angleInDegrees=True)
                opt_flow_values.append(magnitude)
                dlt_flow_values.append(angle)
                gray_frame = gray_frame_next
                
            avg_flow_magnitude = np.mean(opt_flow_values)
            std_flow_angle = np.std(dlt_flow_values)
            print("opt_flow: %s" % avg_flow_magnitude)
            print("dlt_flow: %s" % std_flow_angle)
            opt_flow_windows.append(float(avg_flow_magnitude))
            dlt_flow_windows.append(float(std_flow_angle))
            cap.release()
            cv2.destroyAllWindows()
            gc.collect()
            removal_path = "time_slice.mp4"
            if os.path.exists(removal_path):
                os.remove(removal_path)
        #print(opt_flow_windows)
        #print(dlt_flow_windows)
        # write to file
        print("\nwriting flow values\n")
        f = open("%s_analysis/FLOWvalues_%s.txt" % (inp,filename[:-4]), "w")
        f.write("optical_flow\tdelta_flow\n")
        for i in range(len(opt_flow_windows)):
            print("%s\t%s\t%s" % (i,opt_flow_windows[i], dlt_flow_windows[i]))
            f.write("%s\t%s\n" % (opt_flow_windows[i], dlt_flow_windows[i]))
        f.close()
      
        print("\ncompleted dense optical flow (Farneback) on %s\n" % filename)



def CESmap():
    print("collecting CES data for %s" % inp)
    f1 = open("%s_analysis/CONTRASTvalues_%s.txt" % (inp,inp), "r")
    f2 = open("%s_analysis/FLOWvalues_%s.txt" % (inp,inp), "r")
    lines_f1 = f1.readlines()
    lines_f2 = f2.readlines()
    # write file
    vals_control = []
    vals_energy = []
    vals_surprise = []
    f = open("%s_analysis/ternary_video_raw.txt" % (inp), "w")
    f.write("energy,control,surprise\n")
    for i in range(len(lines_f1)):
        if(i==0):
            continue
        array_f1 = lines_f1[i].split()
        control = array_f1[0]
        array_f2 = lines_f2[i].split()
        energy = array_f2[0]
        surprise = array_f2[1]
        vals_control.append(float(control))
        vals_energy.append(float(energy))
        vals_surprise.append(float(surprise))
        print("%s\t%s\t%s\t%s" % (i,control, energy, surprise))
        f.write("%s,%s,%s\n" % (energy, control, surprise))
    f.close()
    # self-normalize each feature column
    if(selfOpt == "yes"):
        vals_control = np.array(vals_control)
        vals_energy = np.array(vals_energy)
        vals_surprise = np.array(vals_surprise)
        norm_control = (vals_control - np.min(vals_control)) / (np.max(vals_control) - np.min(vals_control))
        norm_energy = (vals_energy - np.min(vals_energy)) / (np.max(vals_energy) - np.min(vals_energy))
        norm_surprise = (vals_surprise- np.min(vals_surprise)) / (np.max(vals_surprise) - np.min(vals_surprise))
    # z-normalize each feature column to underwater ambient scenery
    if(selfOpt == "no"):
        vals_control = np.array(vals_control)
        vals_energy = np.array(vals_energy)
        vals_surprise = np.array(vals_surprise)
        control_mean = 52.239
        control_sd = 1.502
        energy_mean = 0.169
        energy_sd = 0.007
        surprise_mean = 102.098
        surprise_sd = 0.198
        norm_control = (vals_control - control_mean) / (control_sd)
        norm_energy = (vals_energy - energy_mean) / (energy_sd)
        norm_surprise = (vals_surprise - surprise_mean) / (surprise_sd)
    f = open("%s_analysis/ternary_video_norm.txt" % (inp), "w")
    f.write("energy,control,surprise\n")
    for i in range(len(norm_control)):
        if(i==0):
            continue
        control = norm_control[i]
        energy = norm_energy[i]
        surprise = norm_surprise[i]
        # normalize each row for ternary plot
        t_control = control/(control+energy+surprise)
        t_energy = energy/(control+energy+surprise)
        t_surprise = surprise/(control+energy+surprise)
        print("%s\t%s\t%s\t%s" % (i,t_control, t_energy, t_surprise))
        f.write("%s,%s,%s\n" % (t_energy, t_control, t_surprise))
    f.close()


def CESmap_batch():
    lst = os.listdir(inp) # your directory path
    number_files = len(lst)
    print("number of files")
    print(number_files)
    dir_list = os.listdir(inp)
    print(dir_list)
    for i in range(number_files):    
        # Open an mp3 file 
        filename = dir_list[i]
        print("collecting CES data for %s" % filename)
        f1 = open("%s_analysis/CONTRASTvalues_%s.txt" % (inp,filename[:-4]), "r")
        f2 = open("%s_analysis/FLOWvalues_%s.txt" % (inp,filename[:-4]), "r")
        lines_f1 = f1.readlines()
        lines_f2 = f2.readlines()
        # write file
        vals_control = []
        vals_energy = []
        vals_surprise = []
        f = open("%s_analysis/ternary_video_raw_%s.txt" % (inp,filename[:-4]), "w")
        f.write("energy,control,surprise\n")
        for i in range(len(lines_f1)):
            if(i==0):
                continue
            array_f1 = lines_f1[i].split()
            control = array_f1[0]
            array_f2 = lines_f2[i].split()
            energy = array_f2[0]
            surprise = array_f2[1]
            vals_control.append(float(control))
            vals_energy.append(float(energy))
            vals_surprise.append(float(surprise))
            print("%s\t%s\t%s\t%s" % (i,control, energy, surprise))
            f.write("%s,%s,%s\n" % (energy, control, surprise))
        f.close()
        # self-normalize each feature column
        if(selfOpt == "yes"):
            vals_control = np.array(vals_control)
            vals_energy = np.array(vals_energy)
            vals_surprise = np.array(vals_surprise)
            norm_control = (vals_control - np.min(vals_control)) / (np.max(vals_control) - np.min(vals_control))
            norm_energy = (vals_energy - np.min(vals_energy)) / (np.max(vals_energy) - np.min(vals_energy))
            norm_surprise = (vals_surprise- np.min(vals_surprise)) / (np.max(vals_surprise) - np.min(vals_surprise))
        # z-normalize each feature column to underwater ambient scenery
        if(selfOpt == "no"):
            vals_control = np.array(vals_control)
            vals_energy = np.array(vals_energy)
            vals_surprise = np.array(vals_surprise)
            control_mean = 52.239
            control_sd = 1.502
            energy_mean = 0.169
            energy_sd = 0.007
            surprise_mean = 102.098
            surprise_sd = 0.198
            norm_control = (vals_control - control_mean) / (control_sd)
            norm_energy = (vals_energy - energy_mean) / (energy_sd)
            norm_surprise = (vals_surprise - surprise_mean) / (surprise_sd)
        f = open("%s_analysis/ternary_video_norm_%s.txt" % (inp,filename[:-4]), "w")
        f.write("energy,control,surprise\n")
        for i in range(len(norm_control)):
            if(i==0):
                continue
            control = norm_control[i]
            energy = norm_energy[i]
            surprise = norm_surprise[i]
            # normalize each row for ternary plot
            t_control = control/(control+energy+surprise)
            t_energy = energy/(control+energy+surprise)
            t_surprise = surprise/(control+energy+surprise)
            print("%s\t%s\t%s\t%s" % (i,t_control, t_energy, t_surprise))
            f.write("%s,%s,%s\n" % (t_energy, t_control, t_surprise))
        f.close()
        
        
def createSoundFiles_batch():
    lst = os.listdir(inp) # your directory path
    number_files = len(lst)
    print("number of files")
    print(number_files)
    dir_list = os.listdir(inp)
    print(dir_list)
    for i in range(number_files):    
        # Open an mp3 file 
        filename = dir_list[i]
        print("converting to .wav format for %s" % inp) 
        song = AudioSegment.from_file(filename, format="mp4") 
        my_path = "%s_analysis/%s.wav" % (inp,filename[:-4])
        song.export(my_path, format="wav")
        
#################################################################################
####################  main program      #########################################
#################################################################################
def main():
    if(fileORfolder == "file"):
        # Path to your MP4 file
        fpath = "%s" % input1
        file_size_bytes = os.path.getsize(fpath) # Get size in bytes
        file_size_mb = file_size_bytes / (1024 * 1024)# Convert to Megabytes (MB)
        print(f"File Size: {file_size_mb:.2f} MB")
        time.sleep(2)
        GlobOptContrast()
        if(file_size_mb <= 2):
            OptFlow()
        if(file_size_mb > 2):                
            OptFlow2()
        CESmap()
        print("\nvideo processing is complete\n")
        
    if(fileORfolder == "folder"):
        #createSoundFiles_batch()
        GlobOptContrast_batch()
        OptFlow_batch()
        CESmap_batch()
        print("\nvideo processing is complete\n")
###############################################################
if __name__ == '__main__':
    main()
    
    
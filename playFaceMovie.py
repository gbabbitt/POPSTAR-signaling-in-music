#!/usr/bin/env python

#############################################################################
######   POPSTAR software for detecting fitness signaling in music
######   produced by Dr. Gregory A. Babbitt
######   and students at the Rochester Instituteof Technology in 2025.
######   Offered freely without guarantee.  License under GPL v3.0
#############################################################################

from PyQt5 import QtCore, QtGui, QtWidgets
import time
import os
import sys
import platform

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
    if(header == "input"):
        fof = value
        print("file or folder is",fof)    
 ###### variable assignments ######
inp = ""+name+""
tm = int(tm)
fof = ""+fof+""
if(fof=="folder"):
    print("PLAYBACK BUTTONS ONLY WORK FOR SINGLE FILE SUBMISSIONS")
    print("movies generated can be found in your main folder")
    exit()
# calculate number of faces for single file
lst = os.listdir("%s_analysis/intervals/" % inp) # your directory path
frame_num = int(len(lst)/4)  # note folder has 4 types of files
print("number of movie frames is %s" % frame_num)
# calculate number of faces for folder

def playFace(myfile):
    if(current_os == "Linux"):
        print("playing movie file (Linux)")
        cmd = "celluloid %s" % myfile
        os.system(cmd)
    if(current_os == "Windows"):
        print("playing movie file (Windows)")
        ## try this
        cmd = "C:\Program Files\Windows Media Player\wmplayer.exe %s" % myfile
        os.system(cmd)
        ## or this
        #from os import startfile
        #startfile(myfile)
    if(current_os == "macOS"):
        print("playing movie file (macOS)")
        cmd = "open ~/%s" % myfile
        os.system(cmd)    
        
def detect_os():
    os_name = platform.system()
    if os_name == "Windows":
        return "Windows"
    elif os_name == "Darwin":
        return "macOS"
    elif os_name == "Linux":
        return "Linux"
    else:
        return "Unknown"
    
###############################################################
if __name__ == '__main__':
    current_os = detect_os()
    print(f"Operating System: {current_os}")
    myfile = os.path.join(os.path.dirname(__file__), "popstar_results/myMovie_faces_%s.mp4" % inp)
    if os.path.exists(myfile):
        playFace(myfile)
    
    
    
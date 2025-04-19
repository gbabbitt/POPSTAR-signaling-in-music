#!/usr/bin/env python

#############################################################################
######   POPSTAR software for detecting fitness signaling in music
######   produced by Dr. Gregory A. Babbitt
######   and students at the Rochester Instituteof Technology in 2022.
######   Offered freely without guarantee.  License under GPL v3.0
#############################################################################

from PyQt5 import QtCore, QtGui, QtWidgets
import time
import os
import sys


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

def playFace(myfile):
    print("playing movie file")
    cmd = "celluloid %s" % myfile
    os.system(cmd)

###############################################################
if __name__ == '__main__':
    import os
    import sys
    myfile = os.path.join(os.path.dirname(__file__), "myMovie_faces.mp4")
    if os.path.exists(myfile):
        playFace(myfile)
    
    
    
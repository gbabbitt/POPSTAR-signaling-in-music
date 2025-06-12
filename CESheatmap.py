#!/usr/bin/env python

#############################################################################
######   ATOMDANCE software suite for machine-learning assisted
######   comparative protein dynamics produced by Dr. Gregory A. Babbitt
######   and students at the Rochester Instituteof Technology in 2022.
######   Offered freely without guarantee.  License under GPL v3.0
#############################################################################

import getopt, sys # Allows for command line arguments
import os
import random as rnd
#import pytraj as pt
import re

# for ggplot
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from plotnine import *
import statsmodels.api as sm
import seaborn as sn 
import csv

###############################################################################
###############################################################################
features = ['energy-AC1','energy-AMP','energy-TEMPO','control-EVI','control-FFV','control-HEN','surprise-LZC','surprise-MSE','surprise-NVI']

# build list of input files
lst = []
dir_list = os.listdir("popstar_results/")
#print(dir_list)
for i in range(len(dir_list)):
    myFile = dir_list[i]
    print(myFile[0:9])
    if(myFile[0:9] == "stats_CFA"):
        lst.append(str(myFile))
print(lst)

##############################################################################
###############################################################################
def collectDF():
    print("collecting dataframe")
    writePath = "popstar_results/CES_signal.txt"
    txt_out = open(writePath, 'w')
    #txt_out.write("folder\tCES\tvalue\n")
    labels = []
    number_files = len(lst)
    print("number of files")
    print(number_files)
    for k in range(number_files):
        inpFile = lst[k]
        readPath = "popstar_results/%s" % inpFile
        print("reading file %s\n" % inpFile)
        with open(readPath, "r") as file:
            lines = file.readlines()
            for line in lines:
                line_array = line.split()
                #print(line_array)
                if(len(line_array) == 0):
                    continue
                if(str(line_array[0]) == "factor"):
                    myFolder = line_array[3]
                    print("folder is %s" % myFolder)
                    labels.append(myFolder)
                matches = re.findall(r"[-+]?\d*\.\d+|\d+", line)
                floats = []
                for match in matches:
                    try:
                        floats.append(float(match))
                    except ValueError:
                        print("Could not convert to float")
                if(len(floats)==3):
                    print("my floats are %s" % floats)
                    floats = np.array(floats)
                    if(floats[0] != 0):
                        myType = "energy"
                    if(floats[1] != 0):
                        myType = "control"
                    if(floats[2] != 0):
                        myType = "surprise"    
                    non_zero_floats = floats[floats != 0]
                    myValue = non_zero_floats[0]
                    print("%s\t%s\t%s\n" % (myFolder, myType, myValue))
                    txt_out.write("%s\t%s\t%s\n" % (myFolder, myType, myValue))
    txt_out.close()
    print("CES_signal.txt is completed")
    global comparisons
    comparisons = labels

def matrix_maker():
    writePath = "popstar_results/CES_signal_matrix.txt"
    txt_out = open(writePath, 'w')
    readPath = "popstar_results/CES_signal.txt"
    with open(readPath, "r") as file:
        lines = file.readlines()
        cnt = 0
        for line in lines:
            cnt = cnt+1
            line_array = line.split()
            print(line_array)
            myValue = line_array[2]
            print("%s\t" % myValue)
            if(cnt < 9):
                txt_out.write("%s\t" % myValue)
            if(cnt == 9):
                print("%s\n" % myValue)
                txt_out.write("%s\n" % myValue)
                cnt = 0
    txt_out.close()
    print("\nmatrix is done\n")
    
def heat_map():
    readPath = "popstar_results/CES_signal_matrix.txt"
    
    with open(readPath, 'r') as file:
        csv_reader = csv.reader(file, delimiter='\t')
        txt_in = list(csv_reader)
    
    print(txt_in)
    txt_in = np.array(txt_in)
    txt_in = txt_in.astype(float)
    txt_in = np.round(txt_in, 2)
    print(txt_in)
    fig, ax = plt.subplots(figsize=(8, 6))
    hm = sn.heatmap(data = txt_in, ax=ax, cmap="rocket", xticklabels = features, yticklabels = comparisons, annot = False)
    ax.set_aspect('equal') # Ensure square cells
    plt.tight_layout()
    plt.savefig('popstar_results/CES_signal_matrix.jpg')
    plt.show()
    plt.close()
    print("\nheatmap is done\n") 


###############################################################
###############################################################
def main():
    collectDF()
    matrix_maker()
    heat_map()
    print("\nheatmap is completed\n")
###############################################################
if __name__ == '__main__':
    main()
    
    
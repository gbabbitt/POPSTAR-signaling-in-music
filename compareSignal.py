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
import seaborn as sns
# for ggplot
from plotnine import *

#######################
import math
import random
bootstp = 50
import random as rnd
#import pytraj as pt
#import nglview as nv
from scipy.spatial import distance
from scipy.stats import entropy
from scipy.stats import ks_2samp, kruskal, f_oneway
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.decomposition import TruncatedSVD
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
import multiprocessing
################################################################################
# find number of cores
num_cores = multiprocessing.cpu_count()
#num_cores = 1 # activate this line for identifying/removing files that stop script with errors
if not os.path.exists('popstar_results'):
        os.mkdir('popstar_results')
# read popstar ctl file
infile = open("popstar-compare.ctl", "r")
infile_lines = infile.readlines()
for x in range(len(infile_lines)):
    infile_line = infile_lines[x]
    #print(infile_line)
    infile_line_array = str.split(infile_line, ",")
    header = infile_line_array[0]
    value = infile_line_array[1]
    #print(header)
    #print(value)
    if(header == "folder1"):
        name1 = value
        print("my file/folder name is",name1)
    if(header == "folder2"):
        name2 = value
        print("my file/folder name is",name2)    
infile.close()
 ###### variable assignments ######
inp1 = ""+name1+""
inp2 = ""+name2+""

print("comparing folders %s and %s" % (inp1,inp2))
#####################################################################
def collectDF():
    print("collecting dataframe")
    writePath = "popstar_results/ternary_compare_%s_%s.txt" % (inp1,inp2)
    writePath1 = "popstar_results/energy_compare_%s_%s.txt" % (inp1,inp2)
    writePath2 = "popstar_results/control_compare_%s_%s.txt" % (inp1,inp2)
    writePath3 = "popstar_results/surprise_compare_%s_%s.txt" % (inp1,inp2)
    txt_out = open(writePath, "w")
    txt_out1 = open(writePath1, "w")
    txt_out2 = open(writePath2, "w")
    txt_out3 = open(writePath3, "w")
    txt_out.write("folder\tternary\tvalue\n")
    txt_out1.write("folder\tenergy\n")
    txt_out2.write("folder\tcontrol\n")
    txt_out3.write("folder\tsurprise\n")
    lst = os.listdir("%s_analysis/intervals/" % (inp1)) # your directory path
    number_files = len(lst)
    print("number of files")
    print(number_files)
    dir_list = os.listdir("%s_analysis/intervals/" % (inp1))
    print(dir_list)
    for fname in dir_list:
        print(fname)
        dirname = fname
        readPath = "%s_analysis/ternary_norm_%s.txt" % (inp1,dirname)
        df = pd.read_csv(readPath, sep = "\t")
        #print(df)
        for i in range(len(df)-1):
            df_row = df.iloc[i,0]
            df_row = df_row.split(",")
            #print(df_row)
            energy = df_row[0]
            control = df_row[1]
            surprise = df_row[2]
            print("%s\t%s\t%s\t%s" % (inp1,energy, control, surprise))
            txt_out.write("%s\tenergy\t%s\n" % (inp1,energy))
            txt_out.write("%s\tcontrol\t%s\n" % (inp1,control))
            txt_out.write("%s\tsurprise\t%s\n" % (inp1,surprise))
            txt_out1.write("%s\t%s\n" % (inp1,energy))
            txt_out2.write("%s\t%s\n" % (inp1,control))
            txt_out3.write("%s\t%s\n" % (inp1,surprise))
    ####################################
    lst = os.listdir("%s_analysis/intervals/" % (inp2)) # your directory path
    number_files = len(lst)
    print("number of files")
    print(number_files)
    dir_list = os.listdir("%s_analysis/intervals/" % (inp2))
    print(dir_list)
    for fname in dir_list:
        print(fname)
        dirname = fname
        readPath = "%s_analysis/ternary_norm_%s.txt" % (inp2,dirname)
        df = pd.read_csv(readPath, sep = "\t")
        #print(df)    
        for i in range(len(df)-1):
            df_row = df.iloc[i,0]
            df_row = df_row.split(",")
            #print(df_row)
            energy = df_row[0]
            control = df_row[1]
            surprise = df_row[2]
            print("%s\t%s\t%s\t%s" % (inp2,energy, control, surprise))
            txt_out.write("%s\tenergy\t%s\n" % (inp2,energy))
            txt_out.write("%s\tcontrol\t%s\n" % (inp2,control))
            txt_out.write("%s\tsurprise\t%s\n" % (inp2,surprise))
            txt_out1.write("%s\t%s\n" % (inp2,energy))
            txt_out2.write("%s\t%s\n" % (inp2,control))
            txt_out3.write("%s\t%s\n" % (inp2,surprise))
    txt_out.close()
    txt_out1.close()
    txt_out2.close()
    txt_out3.close()
    
def KruskalWallis():
    print("Kruskal-Wallis tests on energy, control and surprise")
    readPath = "popstar_results/energy_compare_%s_%s.txt" % (inp1,inp2)
    df = pd.read_csv(readPath, sep = "\t")
    groups = [df['energy'][df['folder'] == g] for g in df['folder'].unique()]
    stat, p = kruskal(*groups)
    global energy_H
    global energy_p
    energy_H = round(stat,2)
    energy_p = round(p,2)
    readPath = "popstar_results/control_compare_%s_%s.txt" % (inp1,inp2)
    df = pd.read_csv(readPath, sep = "\t")
    groups = [df['control'][df['folder'] == g] for g in df['folder'].unique()]
    stat, p = kruskal(*groups)
    global control_H
    global control_p
    control_H = round(stat,2)
    control_p = round(p,2)
    readPath = "popstar_results/surprise_compare_%s_%s.txt" % (inp1,inp2)
    df = pd.read_csv(readPath, sep = "\t")
    groups = [df['surprise'][df['folder'] == g] for g in df['folder'].unique()]
    stat, p = kruskal(*groups)
    global surprise_H
    global surprise_p
    surprise_H = round(stat,2)
    surprise_p = round(p,2)
    print("Kruskal-Wallis tests")
    print("energy| H=%s p=%s" % (energy_H,energy_p))
    print("control| H=%s p=%s" % (control_H,control_p))
    print("surprise| H=%s p=%s" % (surprise_H,surprise_p))
    return energy_H
    return energy_p
    return control_H
    return control_p
    return surprise_H
    return surprise_p
    
def  errorBarPlot(energy_H,energy_p,control_H,control_p,surprise_H,surprise_p):   
    # Create a sample dataframe
    readPath = "popstar_results/ternary_compare_%s_%s.txt" % (inp1,inp2)
    df = pd.read_csv(readPath, sep = "\t")
    # Plotting the bar plot with error bars
    sns.barplot(x='folder', y='value', hue='ternary', data=df, errorbar='ci')

    # Adding labels and title
    plt.xlabel('folder')
    plt.ylabel('normalized feature value')
    plt.suptitle('energy H=%s p=%s | control H=%s p=%s' % (energy_H,energy_p,control_H,control_p))
    plt.title('surprise H=%s p=%s' % (surprise_H,surprise_p))

    # Display the plot
    plt.legend(title='ternary axes')
    plt.savefig("popstar_results/compareSignal_%s_%s.png" % (inp1,inp2))
    plt.show()
    
    
    
    
        
#################################################################################
####################  main program      #########################################
#################################################################################
def main():
    collectDF()
    KruskalWallis()
    errorBarPlot(energy_H,energy_p,control_H,control_p,surprise_H,surprise_p)
    print("\nsignal comparison is complete\n")   
    
        
###############################################################
if __name__ == '__main__':
    main()
    
    
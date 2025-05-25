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
from scipy.stats import ks_2samp, gaussian_kde
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
from scipy.stats import f_oneway
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.decomposition import TruncatedSVD
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
import multiprocessing
################################################################################
# find number of cores
num_cores = multiprocessing.cpu_count()
#num_cores = 1 # activate this line for identifying/removing files that stop script with errors

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
    writePath = "distance_order1_compare_%s_%s.txt" % (inp1,inp2)
    txt_out = open(writePath, "w")
    txt_out.write("folder\tdistance\n")
    lst = os.listdir("%s_analysis/intervals/" % (inp1)) # your directory path
    number_files = len(lst)
    print("number of files")
    print(number_files)
    dir_list = os.listdir("%s_analysis/intervals/" % (inp1))
    print(dir_list)
    for fname in dir_list:
        print(fname)
        dirname = fname
        readPath = "%s_analysis/distances_order1_%s.txt" % (inp1,dirname)
        df = pd.read_csv(readPath, sep = "\t")
        #print(df)
        for i in range(len(df)-1):
            distance = df.iloc[i,0]
            print("%s\t%s" % (inp1,distance))
            txt_out.write("%s\t%s\n" % (inp1,distance))
    ##################################        
    lst = os.listdir("%s_analysis/intervals/" % (inp2)) # your directory path
    number_files = len(lst)
    print("number of files")
    print(number_files)
    dir_list = os.listdir("%s_analysis/intervals/" % (inp2))
    print(dir_list)
    for fname in dir_list:
        print(fname)
        dirname = fname
        readPath = "%s_analysis/distances_order1_%s.txt" % (inp2,dirname)
        df = pd.read_csv(readPath, sep = "\t")
        #print(df)
        for i in range(len(df)-1):
            distance = df.iloc[i,0]
            print("%s\t%s" % (inp2,distance))
            txt_out.write("%s\t%s\n" % (inp2,distance))
    txt_out.close()
    
def KLdiv():
    print("Kullback-Leibler divergence and 2 sample Kolmogorov-Smirnov test")
    readPath = "distance_order1_compare_%s_%s.txt" % (inp1,inp2)
    df = pd.read_csv(readPath, sep = "\t")
    print(df)
    data = df.groupby('folder').agg(list)
    data1 = data.iloc[0,0]
    data2 = data.iloc[1,0]
    data1 = np.array(data1)
    data2 = np.array(data2)
    #print(data1)
    #print(data2)
    
    # Estimate PDFs using Kernel Density Estimation (KDE)
    kde1 = gaussian_kde(data1)
    kde2 = gaussian_kde(data2)

    # Create a range of x values over which to evaluate the PDFs
    x_values = np.linspace(min(data1.min(), data2.min()) - 1, max(data1.max(), data2.max()) + 1, 200)

    pdf1 = kde1(x_values)
    pdf2 = kde2(x_values)
    global p_value
    global kl_div
    global ks_stat
    # Calculate KL divergence
    kl_div = kl_divergence(pdf1, pdf2)
    print(f"KL Divergence: {kl_div}")
    # Perform KS test
    ks_stat, p_value = ks_2samp(data1, data2)
    print(f"KS Statistic: {ks_stat}")
    print(f"P-value: {p_value}")
    return p_value
    return kl_div
    return ks_stat
    
def kl_divergence(p, q):
    #Calculates KL divergence between two probability distributions
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))   
    
def histPlot(p_value, ks_stat, kl_div):
    print("making plot")
    readPath = "distance_order1_compare_%s_%s.txt" % (inp1,inp2)
    df = pd.read_csv(readPath, sep = "\t")
    print(df)
    data = df.groupby('folder').agg(list)
    print(data)
    data1 = data.iloc[0,0]
    data2 = data.iloc[1,0]
    row1 = data.iloc[0]
    row2 = data.iloc[1]
    label1 = row1.name
    label2 = row2.name   
    data1 = np.array(data1)
    data2 = np.array(data2)
    sns.histplot(data=data1, label= "%s" % label1, kde=True) # kde=True adds density curve
    sns.histplot(data=data2, label="%s" % label2, kde=True)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    ks_stat = round(ks_stat, 5)
    p_value = round(p_value, 5)
    plt.title("KS test | D=%s p=%s" % (ks_stat,p_value))
    plt.suptitle("KL divergence = %s" % (kl_div))
    plt.legend()
    plt.savefig("compareDist_%s_%s.png" % (inp1,inp2))
    plt.show()
    
        
#################################################################################
####################  main program      #########################################
#################################################################################
def main():
    collectDF()
    KLdiv()
    histPlot(p_value, ks_stat, kl_div)
    print("\nstep distribution comparison is complete\n")   
    
        
###############################################################
if __name__ == '__main__':
    main()
    
    
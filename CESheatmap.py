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
from scipy.stats import kruskal
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from plotnine import *
import statsmodels.api as sm
import seaborn as sn 
import csv

###############################################################################
###############################################################################
features = ['% stability(CES)']

# build list of input files
lst = []
dir_list = os.listdir("popstar_results/")
#print(dir_list)
for i in range(len(dir_list)):
    myFile = dir_list[i]
    print(myFile[0:9])
    if(myFile[0:11] == "permutation"):
        lst.append(str(myFile))
print(lst)

##############################################################################
###############################################################################
def collectDF():
    print("collecting dataframe")
    writePath = "popstar_results/CES_signal.txt"
    txt_out = open(writePath, 'w')
    #txt_out.write("folder\tp-value\tnonrandom\n")
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
                if(str(line_array[0]) == "empirical"):
                    myPval = line_array[3]
                    myFolder = line_array[5]
                    print("folder is %s" % myFolder)
                    labels.append(myFolder)
                if(str(line_array[0]) == "interpretation"):
                    myNRval = line_array[2]
            # print all files    
            print("%s\t%s\t%s\n" % (myFolder, myPval, myNRval))
            txt_out.write("%s\t%s\t%s\n" % (myFolder, myPval, myNRval))
            
            
    txt_out.close()
    print("CES_signal.txt is completed")
    global comparisons
    comparisons = labels

def matrix_maker_files():
    writePath = "popstar_results/CES_signal_matrix_files.txt"
    txt_out = open(writePath, 'w')
    readPath = "popstar_results/CES_signal.txt"
    with open(readPath, "r") as file:
        lines = file.readlines()
        for line in lines:
            line_array = line.split()
            print(line_array)
            myValue = line_array[2]
            print("%s\n" % myValue)
            txt_out.write("%s\n" % myValue)
    txt_out.close()
    print("\nmatrix is done\n")
    
def heat_map_files():
    readPath = "popstar_results/CES_signal_matrix_files.txt"
    
    with open(readPath, 'r') as file:
        csv_reader = csv.reader(file, delimiter='\t')
        txt_in = list(csv_reader)
    
    print(txt_in)
    txt_in = np.array(txt_in)
    txt_in = txt_in.astype(float)
    txt_in = np.round(txt_in, 2)
    print(txt_in)
    # sort rows ascending
    df = pd.DataFrame(txt_in)
    sorted_rows = df.sum(axis=1).sort_values(ascending=False).index
    df_sorted_rows = df.reindex(index=sorted_rows)
    lb = pd.DataFrame(comparisons)
    sorted_labels = lb.reindex(index=sorted_rows)
    txt_in = df_sorted_rows.to_numpy()
    sorted_labels = sorted_labels[0].values
    # make plot
    fig, ax = plt.subplots(figsize=(8, 6))
    hm = sn.heatmap(data = txt_in, ax=ax, cmap="rocket", xticklabels = features, yticklabels = comparisons, annot = False, vmin = 0, vmax = 100)
    ax.set_aspect('equal') # Ensure square cells
    plt.tight_layout()
    plt.savefig('popstar_results/CES_signal_matrix_files.jpg')
    plt.show()
    plt.close()
    myMIN = np.min(txt_in)
    myMAX = np.max(txt_in)
    fig, ax = plt.subplots(figsize=(8, 6))
    hm = sn.heatmap(data = txt_in, ax=ax, cmap="rocket", xticklabels = features, yticklabels = comparisons, annot = False, vmin = myMIN, vmax = myMAX)
    ax.set_aspect('equal') # Ensure square cells
    plt.tight_layout()
    plt.savefig('popstar_results/CES_signal_matrix_files_scaled.jpg')
    plt.show()
    plt.close()
    print("\nheatmap is done\n") 

def matrix_maker_folders():
    writePath = "popstar_results/CES_signal_matrix_folders.txt"
    readPath = "popstar_results/CES_signal.txt"
    df = pd.read_csv(readPath, sep = "\t", header=None)
    avgs_df = df.groupby(0, as_index=False).mean()
    print(df)
    print(avgs_df)
    avgs_df[2].to_csv(writePath, index=False, header=False)
    global avg_comparisons
    avg_comparisons = avgs_df[0].to_numpy()
    # KW test
    group_data = [group[2].values for name, group in df.groupby(0)]
    global h_statistic
    global p_value
    h_statistic, p_value = kruskal(*group_data)
    print(f"H-statistic: {h_statistic}")
    print(f"P-value: {p_value}")
    h_statistic = round(h_statistic, 3)
    p_value = round(p_value, 3)
    print("\nmatrix is done\n")
    
def heat_map_folders():
    readPath = "popstar_results/CES_signal_matrix_folders.txt"
    with open(readPath, 'r') as file:
        csv_reader = csv.reader(file, delimiter='\t')
        txt_in = list(csv_reader)
    print(txt_in)
    txt_in = np.array(txt_in)
    txt_in = txt_in.astype(float)
    txt_in = np.round(txt_in, 2)
    print(txt_in)
    # sort rows ascending
    df = pd.DataFrame(txt_in)
    sorted_rows = df.sum(axis=1).sort_values(ascending=False).index
    df_sorted_rows = df.reindex(index=sorted_rows)
    lb = pd.DataFrame(avg_comparisons)
    sorted_labels = lb.reindex(index=sorted_rows)
    txt_in = df_sorted_rows.to_numpy()
    sorted_labels = sorted_labels[0].values
    # make plot
    fig, ax = plt.subplots(figsize=(8, 6))
    hm = sn.heatmap(data = txt_in, ax=ax, cmap="rocket", xticklabels = features, yticklabels = sorted_labels, annot = False, vmin = 0, vmax = 100)
    ax.set_aspect('equal') # Ensure square cells
    plt.title('KW test | h=%s, p=%s' % (h_statistic,p_value))
    plt.tight_layout()
    plt.savefig('popstar_results/CES_signal_matrix_folders.jpg')
    plt.show()
    plt.close()
    myMIN = np.min(txt_in)
    myMAX = np.max(txt_in)
    fig, ax = plt.subplots(figsize=(8, 6))
    hm = sn.heatmap(data = txt_in, ax=ax, cmap="rocket", xticklabels = features, yticklabels = sorted_labels, annot = False, vmin = myMIN, vmax = myMAX)
    ax.set_aspect('equal') # Ensure square cells
    plt.title('KW test | h=%s, p=%s' % (h_statistic,p_value))
    plt.tight_layout()
    plt.savefig('popstar_results/CES_signal_matrix_folders_scaled.jpg')
    plt.show()
    plt.close()
    print("\nheatmap is done\n") 

def errorbar_folders():
    readPath = "popstar_results/CES_signal.txt"
    df = pd.read_csv(readPath, sep = "\t", header=None)
    print(df)
    folder = df[0]
    value = df[2]
    sn.barplot(x=folder, y=value, hue=folder, palette='viridis', data=df, errorbar='ci')
    #sn.barplot(x=folder, y=value, hue=value, palette='viridis', data=df)
    # Adding labels and title
    plt.xlabel('folder')
    plt.ylabel('% stability (CES)')
    plt.title('KW test | h=%s, p=%s' % (h_statistic,p_value))
    
    # Display the plot
    plt.savefig("popstar_results/CES_signal_bars.png")
    plt.show()
    print("\nerror chart is done\n") 
###############################################################
###############################################################
def main():
    collectDF()
    matrix_maker_files()
    heat_map_files()
    matrix_maker_folders()
    heat_map_folders()
    errorbar_folders()
    print("\nheatmap is completed\n")
###############################################################
if __name__ == '__main__':
    main()
    
    
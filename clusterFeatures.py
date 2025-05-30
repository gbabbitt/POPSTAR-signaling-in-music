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
from matplotlib.colors import ListedColormap
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
from sklearn.preprocessing import StandardScaler
import multiprocessing
# machine learning Random Forest classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.mixture import GaussianMixture
from mpl_toolkits.mplot3d import Axes3D

################################################################################
# find number of cores
num_cores = multiprocessing.cpu_count()
#num_cores = 1 # activate this line for identifying/removing files that stop script with errors
if not os.path.exists('popstar_results'):
        os.mkdir('popstar_results')
# read popstar ctl file
infile = open("popstar-classify.ctl", "r")
infile_lines = infile.readlines()
num_folders = 0
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
        if(name1!=""):
            num_folders=num_folders+1
    if(header == "folder2"):
        name2 = value
        print("my file/folder name is",name2)
        if(name2!=""):
            num_folders=num_folders+1
    if(header == "folder3"):
        name3 = value
        print("my file/folder name is",name3)
        if(name3!=""):
            num_folders=num_folders+1
    if(header == "folder4"):
        name4 = value
        print("my file/folder name is",name4)
        if(name4!=""):
            num_folders=num_folders+1
    if(header == "folder5"):
        name5 = value
        print("my file/folder name is",name5)
        if(name5!=""):
            num_folders=num_folders+1
    if(header == "folder6"):
        name6 = value
        print("my file/folder name is",name6)
        if(name6!=""):
            num_folders=num_folders+1

infile.close()
 ###### variable assignments ######
inp1 = ""+name1+""
inp2 = ""+name2+""
inp3 = ""+name3+""
inp4 = ""+name4+""
inp5 = ""+name5+""
inp6 = ""+name6+""
print("classifying %s folders" % num_folders)

if(num_folders <=1):
    print("not enough categories for classification (must be 2 or more)\n")
    exit()
if(num_folders == 2):
    folder_list = [inp1,inp2]
if(num_folders == 3):
    folder_list = [inp1,inp2,inp3]
if(num_folders == 4):
    folder_list = [inp1,inp2,inp3,inp4]    
if(num_folders == 5):
    folder_list = [inp1,inp2,inp3,inp4,inp5]    
if(num_folders == 6):
    folder_list = [inp1,inp2,inp3,inp4,inp5,inp6]     
    
##############################################################
def collectDF():
    print("collecting dataframe")
    writePath = "popstar_results/EM_features_%s.txt" % folder_list
    txt_out = open(writePath, "w")
    txt_out.write("folder\tenergy\tcontrol\tsurprise\n")
    
    for i in range(len(folder_list)):
        inp = folder_list[i]
        lst = os.listdir("%s_analysis/intervals/" % (inp)) # your directory path
        number_files = len(lst)
        print("number of files")
        print(number_files)
        dir_list = os.listdir("%s_analysis/intervals/" % (inp))
        print(dir_list)
        for fname in dir_list:
            print(fname)
            dirname = fname
            readPath = "%s_analysis/ternary_%s.txt" % (inp,dirname)
            df = pd.read_csv(readPath, sep = "\t")
            #print(df)
            for i in range(len(df)-1):
                df_row = df.iloc[i,0]
                df_row = df_row.split(",")
                #print(df_row)
                energy = df_row[0]
                control = df_row[1]
                surprise = df_row[2]
                print("%s\t%s\t%s\t%s" % (inp,energy,control,surprise))
                txt_out.write("%s\t%s\t%s\t%s\n" % (inp,energy,control,surprise))
    txt_out.close()
        
def clusterEM():
    print("model-based clustering")    
    readPath = "popstar_results/EM_features_%s.txt" % folder_list
    writePath = "popstar_results/stats_EMclustering_%s.txt" % folder_list
    #readPath = "data_boxplots.dat"
    #writePath = "stats_classifiers.dat"
    outfile = open(writePath, "w")
    df = pd.read_csv(readPath, delimiter='\t',header=0)
    print(df)
    
    
    # Generate sample data (replace with your data loading)
    #np.random.seed(0)
    #data = np.random.rand(100, 3)
    #df = pd.DataFrame(data, columns=['Axis1', 'Axis2', 'Axis3'])
    df = pd.read_csv(readPath, delimiter='\t',header=0)
    data = pd.read_csv(readPath, delimiter='\t',header=0)
    df = pd.DataFrame(data, columns=['energy', 'control', 'surprise'])
    # EM clustering
    n_clusters = num_folders
    gmm = GaussianMixture(n_components=n_clusters, random_state=0)
    gmm.fit(df)
    cluster_labels = gmm.predict(df)
    df['folder'] = cluster_labels

    # 3D Scatter plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    for i in range(n_clusters):
        cluster_data = df[df['folder'] == i]
        ax.scatter(cluster_data['energy'], cluster_data['control'], cluster_data['surprise'],
                   c=colors[i % len(colors)], s=4, label=f'{folder_list[i]}')

    ax.set_xlabel('energy')
    ax.set_ylabel('control')
    ax.set_zlabel('surprise')
    ax.legend()
    plt.title('EM Clustering on mean Energy,Control,Surprise values')
    plt.show()
    

#################################################################################
####################  main program      #########################################
#################################################################################
def main():
    collectDF()
    clusterEM()
    print("\nsignal clustering is complete\n")   
    
        
###############################################################
if __name__ == '__main__':
    main()
    
    
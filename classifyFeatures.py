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
    writePath = "popstar_results/RF_features_%s.txt" % folder_list
    txt_out = open(writePath, "w")
    txt_out.write("folder\tenergy-AC1\tenergy-AMP\tcontrol-BIV\tcontrol-EVI\tcontrol-FFV\tcontrol-HEN\tsurprise-LZC\tsurprise-MSE\tsurprise-NVI\tenergy-TEMPO\n")
    
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
            readPath = "%s_analysis/features_norm_%s.txt" % (inp,dirname)
            df = pd.read_csv(readPath, sep = "\t")
            #print(df)
            for i in range(len(df)-1):
                df_row = df.iloc[i,0]
                df_row = df_row.split(",")
                #print(df_row)
                AC1values = df_row[0]
                AMPvalues = df_row[1]
                BIVvalues = df_row[2]
                EVIvalues = df_row[3]
                FFVvalues = df_row[4]
                HENvalues = df_row[5]
                LZCvalues = df_row[6]
                MSEvalues = df_row[7]
                NVIvalues = df_row[8]
                TEMPOvalues = df_row[9]
                print("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s" % (inp,AC1values,AMPvalues,BIVvalues,EVIvalues,FFVvalues,HENvalues,LZCvalues,MSEvalues,NVIvalues,TEMPOvalues))
                txt_out.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" % (inp,AC1values,AMPvalues,BIVvalues,EVIvalues,FFVvalues,HENvalues,LZCvalues,MSEvalues,NVIvalues,TEMPOvalues))
    txt_out.close()
        
def RFclass():
    print("\nconducting RF (random forest) on %s (10 bootstraps)\n" % folder_list) 
    readPath = "popstar_results/RF_features_%s.txt" % folder_list
    writePath = "popstar_results/stats_RFclassifier_%s.txt" % folder_list
    #readPath = "data_boxplots.dat"
    #writePath = "stats_classifiers.dat"
    outfile = open(writePath, "w")
    acc_vals = []
    feature_scores = []
    X_labels = []
    scaler = MinMaxScaler()
    for i in range(10):
        df = pd.read_csv(readPath, delimiter='\t',header=0)
        #print(df)
        y = df.folder
        X = df.drop('folder', axis=1)
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns) # MinMax scaling
        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        # Create a Random Forest Classifier
        clf = RandomForestClassifier(n_estimators=500, class_weight='balanced')
        # Fit the model to the training data
        clf.fit(X_train, y_train)
        # Make predictions on the test data
        y_pred = clf.predict(X_test)
        # Get feature importances
        feature_importances = clf.feature_importances_
        feature_scores.append(feature_importances)
        X_labels.append(X.columns)
        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        #print("RF accuracy:", accuracy)
        acc_vals.append(accuracy)
    acc_mean = np.average(acc_vals)
    acc_sd = np.std(acc_vals)
    print("\nRF accuracy: %s +- %s\n" % (acc_mean, acc_sd)) 
    outfile.write("\nRFaccuracy: %s +- %s\n" % (acc_mean, acc_sd))    
    #print(feature_importances)
    feature_scores = pd.DataFrame(feature_scores)
    #print(feature_scores)
    feature_means = feature_scores.mean()
    #print(feature_means)
    feature_sem = feature_scores.sem()
    #print(feature_sem)
    #print(X.columns)
    ylim_neg = feature_means - 2*feature_sem
    ylim_pos = feature_means + 2*feature_sem
    #print(feature_means)
    #print(ylim_neg)
    #print(ylim_pos)
    # Create a DataFrame for visualization
    importance_df = pd.DataFrame({"Feature": X.columns, "Importance": feature_means})
    #importance_df = importance_df.sort_values("Importance", ascending=False)
    print(importance_df)
    outfile.write("\nfeature importances:\n")
    outfile.write("\n%s\n" % (importance_df))
    outfile.close
    # Plot feature importances
    #grp_color = ('red','orange','yellow','green','cyan','blue','violet','brown','gray','white','black')
    grp_color = ('red','orange','yellow','green','cyan','blue','violet','brown','gray','white') # option drop order 1 AC
    myplot = (ggplot(importance_df, aes(x='Feature', y='Importance')) + geom_bar(stat = "identity", fill = grp_color) + geom_errorbar(ymin=ylim_neg, ymax=ylim_pos) + labs(title='Feature Importance from Random Forest Model (500 trees, 100 bootstraps)', x='Feature', y='Importance (+- 2 SEM)') + theme(panel_background=element_rect(fill='black', alpha=.2)))
    myplot.save("popstar_results/RF_featureImportance_%s.png" % folder_list, width=10, height=5, dpi=300)
    #myplot.save("data_RF_featureImportance.png", width=10, height=5, dpi=300)
    #myplot.show()
    print(myplot)
    
    

    
    
    
        
#################################################################################
####################  main program      #########################################
#################################################################################
def main():
    collectDF()
    RFclass()
    print("\nsignal classification is complete\n")   
    
        
###############################################################
if __name__ == '__main__':
    main()
    
    
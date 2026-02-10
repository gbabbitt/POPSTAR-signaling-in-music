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
import skfda
from skfda.datasets import fetch_phoneme
from skfda.ml.classification import KNeighborsClassifier

################################################################################
# find number of cores
num_cores = multiprocessing.cpu_count()
#num_cores = 1 # activate this line for identifying/removing files that stop script with errors
if not os.path.exists('popstar_results'):
        os.mkdir('popstar_results')
if not os.path.exists('popstar_results/FDA'):
        os.mkdir('popstar_results/FDA')
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
def findDIM():
    print("finding proper dimension of square grid")
    flengths = []
    fname_cnts = []
    
    #global labels
    labels = np.array([], dtype=int)
    #global Cvals
    Cvals = []
    for j in range(len(folder_list)):
        inp = folder_list[j]
        lst = os.listdir("%s_analysis/intervals/" % (inp)) # your directory path
        number_files = len(lst)
        print("number of files")
        print(number_files)
        dir_list = os.listdir("%s_analysis/intervals/" % (inp))
        print(dir_list)
        if not os.path.exists('popstar_results/FDA/%s' % inp):
            os.mkdir('popstar_results/FDA/%s' % inp)
        fname_cnt = 0
        for fname in dir_list:
            print(fname)
            fname_cnt = fname_cnt+1
            dirname = fname
            readPath = "%s_analysis/ternary_norm_%s.txt" % (inp,dirname)
            df = pd.read_csv(readPath, sep = "\t")
            #print(df)
            
            Cset = []
            for i in range(len(df)-1):
                df_row = df.iloc[i,0]
                df_row = df_row.split(",")
                #print(df_row)
                energy = df_row[0]
                control = df_row[1]
                surprise = df_row[2]
            flengths.append(i)
        fname_cnts.append(fname_cnt)
    print(flengths)
    print(fname_cnts)
    global min_fnames
    global min_flengths
    min_fnames = min(fname_cnts)
    min_flengths = min(flengths)
    print("min fnames = %s" % min_fnames)
    print("min flengths = %s" % min_flengths)
    
def collectDF():
    print("collecting dataframe")
    writePath = "popstar_results/FDA_features_%s.txt" % folder_list
    txt_out = open(writePath, "w")
    txt_out.write("folder\tfile\tenergy\tcontrol\tsurprise\n")
    global labels
    labels = np.array([], dtype=int)
    global Cvals
    Cvals = []
    global Evals
    Evals = []
    global Svals
    Svals = []
    for j in range(len(folder_list)):
        inp = folder_list[j]
        lst = os.listdir("%s_analysis/intervals/" % (inp)) # your directory path
        number_files = len(lst)
        print("number of files")
        print(number_files)
        dir_list = os.listdir("%s_analysis/intervals/" % (inp))
        print(dir_list)
        if not os.path.exists('popstar_results/FDA/%s' % inp):
            os.mkdir('popstar_results/FDA/%s' % inp)
        fname_cnt = 0
        for fname in dir_list:
            print(fname)
            fname_cnt = fname_cnt+1
            dirname = fname
            readPath = "%s_analysis/ternary_%s.txt" % (inp,dirname)
            df = pd.read_csv(readPath, sep = "\t")
            #print(df)
            writePath2 = "popstar_results/FDA/%s/FDA_features_%s.txt" % (inp,dirname)
            txt_out2 = open(writePath2, "w")
            txt_out2.write("folder\tfile\tenergy\tcontrol\tsurprise\n")
            Cset = []
            Eset = []
            Sset = []
            for i in range(len(df)-1):
                df_row = df.iloc[i,0]
                df_row = df_row.split(",")
                #print(df_row)
                energy = df_row[0]
                control = df_row[1]
                surprise = df_row[2]
                cleaned_dirname = "".join(dirname.split()) # remove all whitespace
                print("%s\t%s\t%s\t%s\t%s" % (inp,cleaned_dirname,energy,control,surprise))
                txt_out.write("%s\t%s\t%s\t%s\t%s\n" % (inp,cleaned_dirname,energy,control,surprise))
                txt_out2.write("%s\t%s\t%s\t%s\t%s\n" % (inp,cleaned_dirname,energy,control,surprise))
                if(i<=min_flengths): # max number of sliding window frames
                    Cset.append(control)
                    Eset.append(energy)
                    Sset.append(surprise)
                #correct_labels.append(j)
            if(fname_cnt<=min_fnames): # max number of songs
                Cvals.append(Cset)
                Evals.append(Eset)
                Svals.append(Sset)
                labels = np.append(labels, j)
            txt_out2.close()        
    txt_out.close()

def clusterFDA():       
    print("functional data analysis (FDA)")
    writePath = "popstar_results/FDA_classifier_results_%s.txt" % folder_list
    global txt_out3
    txt_out3 = open(writePath, "w")
    txt_out3.write("FDA results\n\n")
    
    #print(labels)
    
    #### control data ####
    #print(Cvals)
    np_Cvals = np.array(Cvals)
    np_Cvals = np_Cvals.astype(float)
    #print(np_Cvals)
    control_data = skfda.FDataGrid(data_matrix=np_Cvals)
    #print(control_data)
    input_data = control_data
    input_label = "control_data"
    txt_out3.write("\n\nCONTROL\n")
    FDA(input_data,input_label)
        
    #### energy data ####
    #print(Evals)
    np_Evals = np.array(Evals)
    np_Evals = np_Evals.astype(float)
    #print(np_Evals)
    energy_data = skfda.FDataGrid(data_matrix=np_Evals)
    #print(energy_data)
    input_data = energy_data
    input_label = "energy_data"
    txt_out3.write("\n\nENERGY\n")
    FDA(input_data,input_label)
    
    #### surprise data ####
    #print(Svals)
    np_Svals = np.array(Svals)
    np_Svals = np_Svals.astype(float)
    #print(np_Svals)
    surprise_data = skfda.FDataGrid(data_matrix=np_Svals)
    #print(surprise_data)
    input_data = surprise_data
    input_label = "surprise_data"
    txt_out3.write("\n\nSURPRISE\n")
    FDA(input_data,input_label)
    
    #txt_out3.close()

def FDA(input_data, input_label):    
    
    X = input_data
    y = labels
    
        
    # pick only first 2
    #X = X[(y == 0) | (y == 1)]
    #y = y[(y == 0) | (y == 1)]

    n_points = min_flengths

    new_points = X.grid_points[0][:n_points]
    new_data = X.data_matrix[:, :n_points]

    X = X.copy(
        grid_points=new_points,
        data_matrix=new_data,
        domain_range=(float(np.min(new_points)), float(np.max(new_points))),
    )
    # show only 20 functions 
    n_plot = num_folders*min_fnames
    #X[:n_plot].plot(group=y)
    X[:n_plot].plot()
    plt.title("%s (input from each track)" % input_label, loc="left")
    plt.savefig("popstar_results/FDA_input_tracks_%s_%s.png" % (folder_list, input_label))
    plt.show()
       
    
    print("FDA - spline smoothing")
    from skfda.misc.hat_matrix import NadarayaWatsonHatMatrix
    from skfda.misc.kernels import normal, uniform, epanechnikov
    from skfda.preprocessing.smoothing import KernelSmoother

    smoother = KernelSmoother(
        NadarayaWatsonHatMatrix(
            bandwidth=0.05,
            kernel=normal,
        ),
    )
    X_smooth = smoother.fit_transform(X)
    
    fig = X_smooth[:n_plot].plot(group=y)
    
    print("FDA - registration (alignment)")
    from skfda.preprocessing.registration import FisherRaoElasticRegistration
    reg = FisherRaoElasticRegistration(
        penalty=0.01,
    )
    
    if(num_folders==2):
       X_reg_grp1 = reg.fit_transform(X_smooth[:n_plot][y[:n_plot] == 0])
       #fig = X_reg_grp1.plot(color="C0")
       X_reg_grp1.mean().plot(fig=fig, color="blue", linewidth=3)
       X_reg_grp2 = reg.fit_transform(X_smooth[:n_plot][y[:n_plot] == 1])
       #fig = X_reg_grp2.plot(color="C1")
       X_reg_grp2.mean().plot(fig=fig, color="orange", linewidth=3)
       fd1 = X_reg_grp1.mean()
       fd2 = X_reg_grp2.mean()
       y1 = fd1.data_matrix.flatten()
       y2 = fd2.data_matrix.flatten()
       diff_array = np.abs(y1 - y2)
       difference = np.sum(diff_array)
       print("abs functional difference = %s" % difference)
       txt_out3.write("abs functional difference = %s\n" % difference)
    if(num_folders==3):   
       X_reg_grp1 = reg.fit_transform(X_smooth[:n_plot][y[:n_plot] == 0])
       #fig = X_reg_grp1.plot(color="C0")
       X_reg_grp1.mean().plot(fig=fig, color="blue", linewidth=3)
       X_reg_grp2 = reg.fit_transform(X_smooth[:n_plot][y[:n_plot] == 1])
       #fig = X_reg_grp2.plot(color="C1")
       X_reg_grp2.mean().plot(fig=fig, color="orange", linewidth=3)
       X_reg_grp3 = reg.fit_transform(X_smooth[:n_plot][y[:n_plot] == 2])
       #fig = X_reg_grp3.plot(color="C2")
       X_reg_grp3.mean().plot(fig=fig, color="green", linewidth=3)
       fd1 = X_reg_grp1.mean()
       fd2 = X_reg_grp2.mean()
       fd3 = X_reg_grp3.mean()
       y1 = fd1.data_matrix.flatten()
       y2 = fd2.data_matrix.flatten()
       y3 = fd3.data_matrix.flatten()
       diff_array1 = np.abs(y1-y2)
       diff_array2 = np.array(y2-y3)
       diff_array3 = np.array(y1-y3)
       difference = (np.sum(diff_array1)+np.sum(diff_array2)+np.sum(diff_array3))/3
       print("abs functional difference = %s" % difference)
       txt_out3.write("abs functional difference = %s\n" % difference)
    if(num_folders==4):   
       X_reg_grp1 = reg.fit_transform(X_smooth[:n_plot][y[:n_plot] == 0])
       #fig = X_reg_grp1.plot(color="C0")
       X_reg_grp1.mean().plot(fig=fig, color="blue", linewidth=3)
       X_reg_grp2 = reg.fit_transform(X_smooth[:n_plot][y[:n_plot] == 1])
       #fig = X_reg_grp2.plot(color="C1")
       X_reg_grp2.mean().plot(fig=fig, color="orange", linewidth=3)
       X_reg_grp3 = reg.fit_transform(X_smooth[:n_plot][y[:n_plot] == 2])
       #fig = X_reg_grp3.plot(color="C2")
       X_reg_grp3.mean().plot(fig=fig, color="green", linewidth=3)
       X_reg_grp4 = reg.fit_transform(X_smooth[:n_plot][y[:n_plot] == 3])
       #fig = X_reg_grp4.plot(color="C3")
       X_reg_grp4.mean().plot(fig=fig, color="red", linewidth=3)
       fd1 = X_reg_grp1.mean()
       fd2 = X_reg_grp2.mean()
       fd3 = X_reg_grp3.mean()
       fd4 = X_reg_grp4.mean()
       y1 = fd1.data_matrix.flatten()
       y2 = fd2.data_matrix.flatten()
       y3 = fd3.data_matrix.flatten()
       y4 = fd4.data_matrix.flatten()
       diff_array1 = np.abs(y1-y2)
       diff_array2 = np.array(y2-y3)
       diff_array3 = np.array(y3-y4)
       diff_array4 = np.array(y1-y3)
       diff_array5 = np.array(y1-y4)
       difference = (np.sum(diff_array1)+np.sum(diff_array2)+np.sum(diff_array3)+np.sum(diff_array4)+np.sum(diff_array5))/5
       print("abs functional difference = %s" % difference)
       txt_out3.write("abs functional difference = %s\n" % difference)
    if(num_folders==5):   
       X_reg_grp1 = reg.fit_transform(X_smooth[:n_plot][y[:n_plot] == 0])
       #fig = X_reg_grp1.plot(color="C0")
       X_reg_grp1.mean().plot(fig=fig, color="blue", linewidth=3)
       X_reg_grp2 = reg.fit_transform(X_smooth[:n_plot][y[:n_plot] == 1])
       #fig = X_reg_grp2.plot(color="C1")
       X_reg_grp2.mean().plot(fig=fig, color="orange", linewidth=3)
       X_reg_grp3 = reg.fit_transform(X_smooth[:n_plot][y[:n_plot] == 2])
       #fig = X_reg_grp3.plot(color="C2")
       X_reg_grp3.mean().plot(fig=fig, color="green", linewidth=3)
       X_reg_grp4 = reg.fit_transform(X_smooth[:n_plot][y[:n_plot] == 3])
       #fig = X_reg_grp4.plot(color="C3")
       X_reg_grp4.mean().plot(fig=fig, color="red", linewidth=3)
       X_reg_grp5 = reg.fit_transform(X_smooth[:n_plot][y[:n_plot] == 4])
       #fig = X_reg_grp5.plot(color="C4")
       X_reg_grp5.mean().plot(fig=fig, color="purple", linewidth=3)
       fd1 = X_reg_grp1.mean()
       fd2 = X_reg_grp2.mean()
       fd3 = X_reg_grp3.mean()
       fd4 = X_reg_grp4.mean()
       fd5 = X_reg_grp5.mean()
       y1 = fd1.data_matrix.flatten()
       y2 = fd2.data_matrix.flatten()
       y3 = fd3.data_matrix.flatten()
       y4 = fd4.data_matrix.flatten()
       y5 = fd5.data_matrix.flatten()
       diff_array1 = np.abs(y1-y2)
       diff_array2 = np.array(y2-y3)
       diff_array3 = np.array(y3-y4)
       diff_array4 = np.array(y4-y5)
       diff_array5 = np.array(y1-y3)
       diff_array6 = np.array(y2-y4)
       diff_array7 = np.array(y1-y4)
       diff_array8 = np.array(y1-y5)
       diff_array9 = np.array(y3-y5)
       difference = (np.sum(diff_array1)+np.sum(diff_array2)+np.sum(diff_array3)+np.sum(diff_array4)+np.sum(diff_array5)+np.sum(diff_array6)+np.sum(diff_array7)+np.sum(diff_array8)+np.sum(diff_array9))/9
       print("abs functional difference = %s" % difference)
       txt_out3.write("abs functional difference = %s\n" % difference)
    if(num_folders==6):   
       X_reg_grp1 = reg.fit_transform(X_smooth[:n_plot][y[:n_plot] == 0])
       #fig = X_reg_grp1.plot(color="C0")
       X_reg_grp1.mean().plot(fig=fig, color="blue", linewidth=3)
       X_reg_grp2 = reg.fit_transform(X_smooth[:n_plot][y[:n_plot] == 1])
       #fig = X_reg_grp2.plot(color="C1")
       X_reg_grp2.mean().plot(fig=fig, color="orange", linewidth=3)
       X_reg_grp3 = reg.fit_transform(X_smooth[:n_plot][y[:n_plot] == 2])
       #fig = X_reg_grp3.plot(color="C2")
       X_reg_grp3.mean().plot(fig=fig, color="green", linewidth=3)
       X_reg_grp4 = reg.fit_transform(X_smooth[:n_plot][y[:n_plot] == 3])
       #fig = X_reg_grp4.plot(color="C3")
       X_reg_grp4.mean().plot(fig=fig, color="red", linewidth=3)
       X_reg_grp5 = reg.fit_transform(X_smooth[:n_plot][y[:n_plot] == 4])
       #fig = X_reg_grp5.plot(color="C4")
       X_reg_grp5.mean().plot(fig=fig, color="red", linewidth=3)
       X_reg_grp6 = reg.fit_transform(X_smooth[:n_plot][y[:n_plot] == 5])
       #fig = X_reg_grp6.plot(color="C5")
       X_reg_grp6.mean().plot(fig=fig, color="brown", linewidth=3)
       fd1 = X_reg_grp1.mean()
       fd2 = X_reg_grp2.mean()
       fd3 = X_reg_grp3.mean()
       fd4 = X_reg_grp4.mean()
       fd5 = X_reg_grp5.mean()
       fd6 = X_reg_grp6.mean()
       y1 = fd1.data_matrix.flatten()
       y2 = fd2.data_matrix.flatten()
       y3 = fd3.data_matrix.flatten()
       y4 = fd4.data_matrix.flatten()
       y5 = fd5.data_matrix.flatten()
       y6 = fd6.data_matrix.flatten()
       diff_array1 = np.abs(y1-y2)
       diff_array2 = np.array(y2-y3)
       diff_array3 = np.array(y3-y4)
       diff_array4 = np.array(y4-y5)
       diff_array5 = np.array(y5-y6)
       diff_array6 = np.array(y1-y3)
       diff_array7 = np.array(y2-y4)
       diff_array8 = np.array(y3-y5)
       diff_array9 = np.array(y4-y6)
       diff_array10 = np.array(y1-y4)
       diff_array11 = np.array(y2-y5)
       diff_array12 = np.array(y3-y6)
       diff_array13 = np.array(y1-y5)
       diff_array14 = np.array(y2-y6)
       diff_array15 = np.array(y1-y6)
       difference = (np.sum(diff_array1)+np.sum(diff_array2)+np.sum(diff_array3)+np.sum(diff_array4)+np.sum(diff_array5)+np.sum(diff_array6)+np.sum(diff_array7)+np.sum(diff_array8)+np.sum(diff_array9)+np.sum(diff_array10)+np.sum(diff_array11)+np.sum(diff_array12)+np.sum(diff_array13)+np.sum(diff_array14)+np.sum(diff_array15))/15
       print("abs functional difference = %s" % difference)
       txt_out3.write("abs functional difference = %s\n" % difference)
    plt.title("%s (smoothed, averaged, and registered classes)" % input_label, loc="left")
    plt.savefig("popstar_results/FDA_classes_%s_%s.png" % (folder_list, input_label))
    plt.show()
    
    print("FDA - functional clustering")
    
    # compile registered and smoothed functional data for train-test split
    if(num_folders==2):
        X_reg = X_reg_grp1.concatenate(X_reg_grp2)
    if(num_folders==3):
        X_reg = X_reg_grp1.concatenate(X_reg_grp2, X_reg_grp3)
    if(num_folders==4):
        X_reg = X_reg_grp1.concatenate(X_reg_grp2, X_reg_grp3, X_reg_grp4)    
    if(num_folders==5):
        X_reg = X_reg_grp1.concatenate(X_reg_grp2, X_reg_grp3, X_reg_grp4, X_reg_grp5) 
    if(num_folders==6):
        X_reg = X_reg_grp1.concatenate(X_reg_grp2, X_reg_grp3, X_reg_grp4, X_reg_grp5, X_reg_grp6) 
    
    
    '''
    print("X_smooth)")
    print(X_smooth)
    print(len(X_smooth))
    print("X_reg)")
    print(X_reg)
    print(len(X_reg))
    print("y")
    print(y)
    print(len(y))
    '''
    
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X_reg,
        y,
        test_size=0.3,
        random_state=0,
        stratify=y,
    )
    
    from skfda.exploratory.depth import ModifiedBandDepth
    from skfda.ml.classification import MaximumDepthClassifier

    depth = MaximumDepthClassifier(depth_method=ModifiedBandDepth())
    depth.fit(X_train, y_train)
    depth_pred = depth.predict(X_test)
    print(depth_pred)
    print(
        f"The score of Maximum Depth Classifier is "
        f"{depth.score(X_test, y_test):2.2%}",
    )
    
    from skfda.ml.classification import KNeighborsClassifier

    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    knn_pred = knn.predict(X_test)
    print(knn_pred)
    print(f"The score of KNN is {knn.score(X_test, y_test):2.2%}")
    
    from skfda.ml.classification import NearestCentroid

    centroid = NearestCentroid()
    centroid.fit(X_train, y_train)
    centroid_pred = centroid.predict(X_test)
    print(centroid_pred)
    print(
        f"The score of Nearest Centroid Classifier is "
        f"{centroid.score(X_test, y_test):2.2%}",
    )
    
    from skfda.exploratory.stats.covariance import ParametricGaussianCovariance
    from skfda.misc.covariances import Gaussian
    from skfda.ml.classification import QuadraticDiscriminantAnalysis

    qda = QuadraticDiscriminantAnalysis(
        ParametricGaussianCovariance(
            Gaussian(variance=6, length_scale=1),
        ),
        regularizer=0.05,
    )
    qda.fit(X_train, y_train)
    qda_pred = qda.predict(X_test)
    print(qda_pred)
    print(f"The score of functional QDA is {qda.score(X_test, y_test):2.2%}")
    print("FDA - plotting signature functions grouped by cluster")
    accuracies = pd.DataFrame({
        "Classification methods":
            [
                "Maximum Depth Classifier",
                "K-Nearest-Neighbors",
                "Nearest Centroid Classifier",
                "Functional QDA",
            ],
        "Accuracy":
            [
                f"{depth.score(X_test, y_test):2.2%}",
                f"{knn.score(X_test, y_test):2.2%}",
                f"{centroid.score(X_test, y_test):2.2%}",
                f"{qda.score(X_test, y_test):2.2%}",
            ],
    })
    
    str_accuracies = str(accuracies)
    print(accuracies)
    txt_out3.write(str_accuracies)
    
    fig, axs = plt.subplots(2, 2)
    plt.subplots_adjust(hspace=0.45, bottom=0.06)

    X_test.plot(group=centroid_pred, axes=axs[0][1])
    axs[0][1].set_title("Nearest Centroid", loc="left")

    X_test.plot(group=depth_pred, axes=axs[0][0])
    axs[0][0].set_title("Maximum Depth", loc="left")

    X_test.plot(group=knn_pred, axes=axs[1][0])
    axs[1][0].set_title("K nearest neighbors", loc="left")

    X_test.plot(group=qda_pred, axes=axs[1][1])
    axs[1][1].set_title("Functional QDA", loc="left")
    plt.suptitle("classifiers - %s" % input_label)
    plt.savefig("popstar_results/FDA_classifiers_%s_%s.png" % (folder_list, input_label))
    plt.show()
    
    
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
    #df['folder'] = cluster_labels
    #print(len(cluster_labels))
    correct_labels_array = np.array(correct_labels)
    #print(len(correct_labels_array))
    print(cluster_labels)
    print(correct_labels_array)
    
    # sort groups by ascending size
    unique_labels, inverse_indices = np.unique(cluster_labels, return_inverse=True)
    group_counts = np.bincount(inverse_indices)
    sorted_indices = np.argsort(group_counts)
    sorted_cluster_labels = unique_labels[sorted_indices]
    print(sorted_cluster_labels)
    unique_labels, inverse_indices = np.unique(correct_labels_array, return_inverse=True)
    group_counts = np.bincount(inverse_indices)
    sorted_indices = np.argsort(group_counts)
    sorted_correct_labels = unique_labels[sorted_indices]
    print(sorted_correct_labels)
    folder_list_sort = []
    for i in range(len(folder_list)):
        myIndex = sorted_correct_labels[i]
        myFolder = folder_list[myIndex]
        folder_list_sort.append(myFolder)
    print(folder_list_sort)
    # calculate % match
    percent_matches = []
    match = 0
    total = 0
    for i in range(len(sorted_correct_labels)):
        test = sorted_cluster_labels[i]
        truth = sorted_correct_labels[i]
        match = 0
        total = 0
        for j in range(len(cluster_labels)-1):
            mytest = cluster_labels[j]
            mytruth = correct_labels[j]
            if(test == mytest and truth == mytruth):
                match = match+1
                total = total+1
            if(test != mytest or truth != mytruth):
                total = total+1
            percent_match = match/total
        percent_matches.append(percent_match)
    print(percent_matches)
    
    
    # 3D Scatter plot
    df['folder'] = cluster_labels
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    for i in range(n_clusters):
        clusterID = i
        folderID = folder_list_sort[i]
        percentID = round(percent_matches[i]*100,2)
        cluster_data = df[df['folder'] == i]
        ax.scatter(cluster_data['energy'], cluster_data['control'], cluster_data['surprise'],
                   c=colors[i % len(colors)], s=4, label=f'{"cluster %s most probable match %s percent to %s" % (clusterID,percentID,folderID)}')
        # just group number
        #ax.scatter(cluster_data['energy'], cluster_data['control'], cluster_data['surprise'],
        #           c=colors[i % len(colors)], s=4, label=f'{"cluster %s" % (clusterID)}')
    
    ax.set_xlabel('energy')
    ax.set_ylabel('control')
    ax.set_zlabel('surprise')
    ax.legend()
    plt.title('EM Clustering of sound fragments on mean Energy,Control,Surprise values')
    plt.show()
    # 3D Scatter plot
    df['folder'] = correct_labels_array
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    for i in range(n_clusters):
        #clusterID = i
        folderID = folder_list[i]
        #percentID = round(percent_matches[i]*100,2)
        cluster_data = df[df['folder'] == i]
        #ax.scatter(cluster_data['energy'], cluster_data['control'], cluster_data['surprise'],
        #           c=colors[i % len(colors)], s=4, label=f'{"cluster %s match %s percent to %s" % (clusterID,percentID,folderID)}')
        ax.scatter(cluster_data['energy'], cluster_data['control'], cluster_data['surprise'],
                   c=colors[i % len(colors)], s=4, label=f'{"folder = %s" % (folderID)}')
    
    ax.set_xlabel('energy')
    ax.set_ylabel('control')
    ax.set_zlabel('surprise')
    ax.legend()
    plt.title('correct labels of sound fragments on mean Energy,Control,Surprise values')
    plt.show()
#################################################################################
####################  main program      #########################################
#################################################################################
def main():
    findDIM()
    collectDF()
    clusterFDA()
    #clusterEM()
    print("\nsignature clustering is complete\n")   
    
        
###############################################################
if __name__ == '__main__':
    main()
    
    
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
from factor_analyzer import (FactorAnalyzer, ConfirmatoryFactorAnalyzer, ModelSpecificationParser)

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
def collectDFrf():
    print("collecting dataframe")
    writePath = "popstar_results/RF_features_%s.txt" % folder_list
    txt_out = open(writePath, "w")
    txt_out.write("folder\tenergy-AC1\tenergy-AMP\tcontrol-BIV\tcontrol-EVI\tcontrol-FFV\tcontrol-HEN\tsurprise-LZC\tsurprise-MSE\tsurprise-NVI\tenergy-TEMPO\n")
    
    for j in range(len(folder_list)):
        inp = folder_list[j]
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
    
def runCFA():   
    print("running confirmatory factor analysis (CFA) on each folder")
    readPath = "popstar_results/RF_features_%s.txt" % folder_list
    writePath = "popstar_results/stats_CFA_%s.txt" % folder_list
    outfile = open(writePath, "w")
    # get data
    df_original = pd.read_csv(readPath, delimiter='\t',header=0)
    #print(df_original)
    
    for j in range(len(folder_list)):
        inp = folder_list[j]
        print("my folder is %s\n" % inp)
        outfile.write("\n-------------------------------------------------\n")
        outfile.write("\n\nmy folder is %s\n\n" % inp)
        # Set the first column as index
        #myIndex = {'folder': ["energy-AC1", "energy-AMP", "energy-TEMPO", "control-EVI", "control-FFV", "control-HEN", "surprise-LZC", "surprise-MSE", "surprise-NVI"]}
        #index_df = pd.DataFrame(myIndex)
        df_indexed = df_original.set_index('folder')
        # Select rows where index (first column) is 'label1'
        df_subset = df_indexed.loc['%s'%inp]
        #print(df_subset)
        energy_AC1 = df_subset['energy-AC1']
        #print(energy_AC1.values)
        energy_AMP = df_subset['energy-AMP']
        #print(energy_AMP.values)
        energy_TEMPO = df_subset['energy-TEMPO']
        #print(energy_TEMPO.values)
        control_EVI = df_subset['control-EVI']
        #print(control_EVI.values)
        control_FFV = df_subset['control-FFV']
        #print(control_FFV.values)
        control_HEN = df_subset['control-HEN']
        #print(control_HEN.values)
        surprise_LZC = df_subset['surprise-LZC']
        #print(surprise_LZC.values)
        surprise_MSE = df_subset['surprise-MSE']
        #print(surprise_MSE.values)
        surprise_NVI = df_subset['surprise-NVI']
        #print(surprise_NVI.values)
        # Define the model
        data  = pd.DataFrame({'energy-AC1': energy_AC1 ,'energy-AMP': energy_AMP,'energy-TEMPO': energy_TEMPO,'control-EVI': control_EVI,'control-FFV': control_FFV,'control-HEN': control_HEN,'surprise-LZC': surprise_LZC,'surprise-MSE': surprise_MSE,'surprise-NVI': surprise_NVI})
        #print(data)
        matrix = pd.DataFrame.cov(data)
        #print(matrix)
        #shuffled_data = data.sample(axis=1, frac=1) # shuffled columns in data frame
        #data = shuffled_data
        #print(myStop)
        ########################################################
        print("\nmodel - hypothesized 3 factor fitness signal = F1:E1,E2,E3 | F2:C1,C2,C3 | F3:S1,S2,S3\n")
        outfile.write("\nmodel - hypothesized 3 factor fitness signal = F1:E1,E2,E3 | F2:C1,C2,C3 | F3:S1,S2,S3\n")
        model_dict = {"F1": ['energy-AC1', 'energy-AMP', 'energy-TEMPO'], "F2": ['control-EVI', 'control-FFV', 'control-HEN'], "F3": ['surprise-LZC', 'surprise-MSE', 'surprise-NVI']}
        
        # using covariance matrix
        model_spec = ModelSpecificationParser.parse_model_specification_from_dict(matrix, model_dict)
        cfa = ConfirmatoryFactorAnalyzer(model_spec, disp=False, n_obs = 9, is_cov_matrix=True)
        cfa.fit(matrix.values)
        
        # using dataframe (gives same result)
        #model_spec = ModelSpecificationParser.parse_model_specification_from_dict(data, model_dict)
        #cfa = ConfirmatoryFactorAnalyzer(model_spec, disp=True, is_cov_matrix=False)
        #cfa.fit(data.values)
        
        # Get the factor loadings
        loadings = cfa.loadings_
        modcov = cfa.get_model_implied_cov()
        factvarcovs = cfa.factor_varcovs_
        se = cfa.get_standard_errors()
        logL = cfa.log_likelihood_
        aic = cfa.aic_
        bic = cfa.bic_
        #print(loadings)
        #outfile.write("\nmodel implied covariance\n")
        #outfile.write(str(modcov))
        #outfile.write("\nfactor variance/covariance\n")
        #outfile.write(str(factvarcovs))
        outfile.write("\nfactor loadings\n")
        outfile.write(str(loadings))
        #outfile.write("\nstandard errors\n")
        #outfile.write(str(se))
        #outfile.write("\nlogL = %s aic= %s bic = %s\n" % (logL,aic,bic))
        outfile.write("\n-------------------------------------------------\n")
        print("Factor Loadings:\n", loadings)
        #print("logL = %s aic= %s bic = %s\n" % (logL,aic,bic))
        ########################################################
        
    outfile.close()
    
def runEFA():   
    print("running exploratory factor analysis (CFA) on each folder")
    readPath = "popstar_results/RF_features_%s.txt" % folder_list
    writePath = "popstar_results/stats_EFA_%s.txt" % folder_list
    outfile = open(writePath, "w")
    # get data
    df_original = pd.read_csv(readPath, delimiter='\t',header=0)
    #print(df_original)
    
    for j in range(len(folder_list)):
        inp = folder_list[j]
        print("my folder is %s\n" % inp)
        outfile.write("\n\nmy folder is %s\n\n" % inp)
        # Set the first column as index
        #myIndex = {'folder': ["energy-AC1", "energy-AMP", "energy-TEMPO", "control-EVI", "control-FFV", "control-HEN", "surprise-LZC", "surprise-MSE", "surprise-NVI"]}
        #index_df = pd.DataFrame(myIndex)
        df_indexed = df_original.set_index('folder')
        # Select rows where index (first column) is 'label1'
        df_indexed = df_indexed.loc['%s'%inp]
        #print(df_indexed)
        # remove index column
        df = df_indexed.iloc[:, 1:]
        # Define the model
        data  = pd.DataFrame({'energy-AC1': df.values[0],'energy-AMP': df.values[1],'energy-TEMPO': df.values[9],'control-EVI': df.values[3],'control-FFV': df.values[4],'control-HEN': df.values[5],'surprise-LZC': df.values[6],'surprise-MSE': df.values[7],'surprise-NVI': df.values[8]})
        
        fa = FactorAnalyzer(n_factors=3, rotation='varimax')
        fa.fit(data)
        # Get the factor loadings
        loadings = fa.loadings_
        coms = fa.get_communalities()
        #print(loadings)
        outfile.write("factor loadings\n")
        outfile.write(str(loadings))
        outfile.write("\ncommunalities\n")
        outfile.write(str(coms))
        print("Factor Loadings:\n", loadings)
        print("Communalities:\n", coms)

def collectDFeb():
    print("collecting dataframe")
    writePath = "popstar_results/ternary_compare_%s.txt" % (folder_list)
    writePath1 = "popstar_results/energy_compare_%s.txt" % (folder_list)
    writePath2 = "popstar_results/control_compare_%s.txt" % (folder_list)
    writePath3 = "popstar_results/surprise_compare_%s.txt" % (folder_list)
    txt_out = open(writePath, "w")
    txt_out1 = open(writePath1, "w")
    txt_out2 = open(writePath2, "w")
    txt_out3 = open(writePath3, "w")
    txt_out.write("folder\tternary\tvalue\n")
    txt_out1.write("folder\tenergy\n")
    txt_out2.write("folder\tcontrol\n")
    txt_out3.write("folder\tsurprise\n")
    
    for j in range(len(folder_list)):
        inp = folder_list[j]
        lst = os.listdir("%s_analysis/intervals/" % (inp)) # your directory path
        number_files = len(lst)
        print("number of files")
        print(number_files)
        dir_list = os.listdir("%s_analysis/intervals/" % (inp))
        print(dir_list)
        for fname in dir_list:
            print(fname)
            dirname = fname
            readPath = "%s_analysis/ternary_norm_%s.txt" % (inp,dirname)
            df = pd.read_csv(readPath, sep = "\t")
            #print(df)
            for i in range(len(df)-1):
                df_row = df.iloc[i,0]
                df_row = df_row.split(",")
                #print(df_row)
                energy = df_row[0]
                control = df_row[1]
                surprise = df_row[2]
                print("%s\t%s\t%s\t%s" % (inp,energy, control, surprise))
                txt_out.write("%s\tenergy\t%s\n" % (inp,energy))
                txt_out.write("%s\tcontrol\t%s\n" % (inp,control))
                txt_out.write("%s\tsurprise\t%s\n" % (inp,surprise))
                txt_out1.write("%s\t%s\n" % (inp,energy))
                txt_out2.write("%s\t%s\n" % (inp,control))
                txt_out3.write("%s\t%s\n" % (inp,surprise))

def  errorBarPlot():   
    # Create a sample dataframe
    readPath = "popstar_results/ternary_compare_%s.txt" % (folder_list)
    df = pd.read_csv(readPath, sep = "\t")
    # Plotting the bar plot with error bars
    sns.barplot(x='folder', y='value', hue='ternary', data=df, errorbar='ci')

    # Adding labels and title
    plt.xlabel('folder')
    plt.ylabel('normalized feature value')
    plt.title('fitness signal comparison')

    # Display the plot
    plt.legend(title='ternary axes')
    plt.savefig("popstar_results/compareSignal_%s.png" % (folder_list))
    plt.show()        
        
#################################################################################
####################  main program      #########################################
#################################################################################
def main():
    collectDFrf()
    collectDFeb()
    runEFA()
    runCFA()
    RFclass()
    errorBarPlot()
    print("\nsignal classification is complete\n")   
    
        
###############################################################
if __name__ == '__main__':
    main()
    
    
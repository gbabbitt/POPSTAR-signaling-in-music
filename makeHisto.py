#!/usr/bin/env python

#############################################################################
######   POPSTAR software for detecting fitness signaling in music
######   produced by Dr. Gregory A. Babbitt
######   and students at the Rochester Instituteof Technology in 2025.
######   Offered freely without guarantee.  License under GPL v3.0
#############################################################################

import getopt, sys # Allows for command line arguments
import os
import random as rnd
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import multiprocessing
# find number of cores
num_cores = multiprocessing.cpu_count()
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
    if(header == "lyrics"):
        lyr = value
        print("lyrics present is",lyr)
       
 ###### variable assignments ######
inp = ""+name+""
tm = int(tm)
fof = ""+fof+""
lyr = ""+lyr+""

def mm_inf(df_order0, df_order1,dirname):
        print("make AIC/BIC")
        test=0
        best=0
        print(df_order1)
        if(fof=="folder"):
            writePath = "%s_analysis/mm_inference_order1_%s.txt" % (inp,dirname)
        if(fof=="file"):
            writePath = "%s_analysis/mm_inference_order1.txt" % (inp)
        txt_out = open(writePath, "w")
        ### gamma function
        fit = sp.stats.gamma.fit(df_order1)
        #print(fit)
        logLik_gamma = np.sum(sp.stats.gamma.logpdf(df_order1, fit[0], loc=fit[1], scale=fit[2]))
        k = len(fit)
        aic_gamma = 2*k - 2*(logLik_gamma)
        bic_gamma = np.log(len(df_order1))*k - 2*(logLik_gamma)
        print("gamma | logLik %s AIC %s BIC %s" % (logLik_gamma, aic_gamma, bic_gamma))
        txt_out.write("gamma | logLik %s AIC %s BIC %s\n" % (logLik_gamma, aic_gamma, bic_gamma))
        if(bic_gamma < test):
            best = bic_gamma
            test = bic_gamma
            bestLABEL = "gamma"
        ### power lognormal function
        fit = sp.stats.powerlognorm.fit(df_order1)
        #print(fit)
        logLik_powerlognorm = np.sum(sp.stats.powerlognorm.logpdf(df_order1, fit[0], fit[1], loc=fit[2], scale=fit[3]))
        k = len(fit)
        aic_powerlognorm = 2*k - 2*(logLik_powerlognorm)
        bic_powerlognorm = np.log(len(df_order1))*k - 2*(logLik_powerlognorm)
        print("powerlognorm | logLik %s AIC %s BIC %s" % (logLik_powerlognorm, aic_powerlognorm, bic_powerlognorm))
        txt_out.write("powerlognorm | logLik %s AIC %s BIC %s\n" % (logLik_powerlognorm, aic_powerlognorm, bic_powerlognorm))
        if(bic_powerlognorm < test):
            best = bic_powerlognorm
            test = bic_powerlognorm
            bestLABEL = "powerlognorm"
        ### lognormal function
        fit = sp.stats.lognorm.fit(df_order1)
        #print(fit)
        logLik_lognorm = np.sum(sp.stats.lognorm.logpdf(df_order1, fit[0], loc=fit[1], scale=fit[2]))
        k = len(fit)
        aic_lognorm = 2*k - 2*(logLik_lognorm)
        bic_lognorm = np.log(len(df_order1))*k - 2*(logLik_lognorm)
        print("lognorm | logLik %s AIC %s BIC %s" % (logLik_lognorm, aic_lognorm, bic_lognorm))
        txt_out.write("lognorm | logLik %s AIC %s BIC %s\n" % (logLik_lognorm, aic_lognorm, bic_lognorm))
        if(bic_lognorm < test):
            best = bic_lognorm
            test = bic_lognorm
            bestLABEL = "lognorm"
        ### pareto function
        fit = sp.stats.pareto.fit(df_order1)
        #print(fit)
        logLik_pareto = np.sum(sp.stats.pareto.logpdf(df_order1, fit[0], loc=fit[1], scale=fit[2]))
        k = len(fit)
        aic_pareto = 2*k - 2*(logLik_pareto)
        bic_pareto = np.log(len(df_order1))*k - 2*(logLik_pareto)
        print("pareto | logLik %s AIC %s BIC %s" % (logLik_pareto, aic_pareto, bic_pareto))
        txt_out.write("pareto | logLik %s AIC %s BIC %s\n" % (logLik_pareto, aic_pareto, bic_pareto))
        if(bic_pareto < test):
            best = bic_pareto
            test = bic_pareto
            bestLABEL = "pareto"
        ### exponential function
        fit = sp.stats.expon.fit(df_order1)
        #print(fit)
        logLik_expon = np.sum(sp.stats.expon.logpdf(df_order1, fit[0], scale=fit[1]))
        k = len(fit)
        aic_expon = 2*k - 2*(logLik_expon)
        bic_expon = np.log(len(df_order1))*k - 2*(logLik_expon)
        print("expon | logLik %s AIC %s BIC %s" % (logLik_expon, aic_expon, bic_expon))
        txt_out.write("expon | logLik %s AIC %s BIC %s\n" % (logLik_expon, aic_expon, bic_expon))
        if(bic_expon < test):
            best = bic_expon
            test = bic_expon
            bestLABEL = "expon"
        ### normal function
        fit = sp.stats.norm.fit(df_order1)
        #print(fit)
        logLik_norm = np.sum(sp.stats.norm.logpdf(df_order1, fit[0], scale=fit[1]))
        k = len(fit)
        aic_norm = 2*k - 2*(logLik_norm)
        bic_norm = np.log(len(df_order1))*k - 2*(logLik_norm)
        print("norm | logLik %s AIC %s BIC %s" % (logLik_norm, aic_norm, bic_norm))
        txt_out.write("norm | logLik %s AIC %s BIC %s\n" % (logLik_norm, aic_norm, bic_norm))
        if(bic_norm < test):
            best = bic_norm
            test = bic_norm
            bestLABEL = "norm"
        ### truncated normal function
        fit = sp.stats.truncnorm.fit(df_order1)
        #print(fit)
        logLik_truncnorm = np.sum(sp.stats.truncnorm.logpdf(df_order1, fit[0], fit[1], loc=fit[2], scale=fit[3]))
        k = len(fit)
        aic_truncnorm = 2*k - 2*(logLik_truncnorm)
        bic_truncnorm = np.log(len(df_order1))*k - 2*(logLik_truncnorm)
        print("truncnorm | logLik %s AIC %s BIC %s" % (logLik_truncnorm, aic_truncnorm, bic_truncnorm))
        txt_out.write("truncnorm | logLik %s AIC %s BIC %s\n" % (logLik_truncnorm, aic_truncnorm, bic_truncnorm))
        if(bic_truncnorm < test):
            best = bic_truncnorm
            test = bic_truncnorm
            bestLABEL = "truncnorm"
        print("BEST MODEL is %s at %s" % (bestLABEL,best))
        txt_out.write("BEST MODEL is %s at %s" % (bestLABEL,best))
        txt_out.close
        ###########################################
        print(df_order0)
        test=0
        best=0
        if(fof=="folder"):
            writePath = "%s_analysis/mm_inference_order0_%s.txt" % (inp,dirname)
        if(fof=="file"):
            writePath = "%s_analysis/mm_inference_order0.txt" % (inp)
        txt_out = open(writePath, "w")
        ### gamma function
        fit = sp.stats.gamma.fit(df_order0)
        #print(fit)
        logLik_gamma = np.sum(sp.stats.gamma.logpdf(df_order0, fit[0], loc=fit[1], scale=fit[2]))
        k = len(fit)
        aic_gamma = 2*k - 2*(logLik_gamma)
        bic_gamma = np.log(len(df_order0))*k - 2*(logLik_gamma)
        print("gamma | logLik %s AIC %s BIC %s" % (logLik_gamma, aic_gamma, bic_gamma))
        txt_out.write("gamma | logLik %s AIC %s BIC %s\n" % (logLik_gamma, aic_gamma, bic_gamma))
        if(bic_gamma < test):
            best = bic_gamma
            test = bic_gamma
            bestLABEL = "gamma"
        ### power lognormal function
        fit = sp.stats.powerlognorm.fit(df_order0)
        #print(fit)
        logLik_powerlognorm = np.sum(sp.stats.powerlognorm.logpdf(df_order0, fit[0], fit[1], loc=fit[2], scale=fit[3]))
        k = len(fit)
        aic_powerlognorm = 2*k - 2*(logLik_powerlognorm)
        bic_powerlognorm = np.log(len(df_order0))*k - 2*(logLik_powerlognorm)
        print("powerlognorm | logLik %s AIC %s BIC %s" % (logLik_powerlognorm, aic_powerlognorm, bic_powerlognorm))
        txt_out.write("powerlognorm | logLik %s AIC %s BIC %s\n" % (logLik_powerlognorm, aic_powerlognorm, bic_powerlognorm))
        if(bic_powerlognorm < test):
            best = bic_powerlognorm
            test = bic_powerlognorm
            bestLABEL = "powerlognorm"
        ### lognormal function
        fit = sp.stats.lognorm.fit(df_order0)
        #print(fit)
        logLik_lognorm = np.sum(sp.stats.lognorm.logpdf(df_order0, fit[0], loc=fit[1], scale=fit[2]))
        k = len(fit)
        aic_lognorm = 2*k - 2*(logLik_lognorm)
        bic_lognorm = np.log(len(df_order0))*k - 2*(logLik_lognorm)
        print("lognorm | logLik %s AIC %s BIC %s" % (logLik_lognorm, aic_lognorm, bic_lognorm))
        txt_out.write("lognorm | logLik %s AIC %s BIC %s\n" % (logLik_lognorm, aic_lognorm, bic_lognorm))
        if(bic_lognorm < test):
            best = bic_lognorm
            test = bic_lognorm
            bestLABEL = "lognorm"
        ### pareto function
        fit = sp.stats.pareto.fit(df_order0)
        #print(fit)
        logLik_pareto = np.sum(sp.stats.pareto.logpdf(df_order0, fit[0], loc=fit[1], scale=fit[2]))
        k = len(fit)
        aic_pareto = 2*k - 2*(logLik_pareto)
        bic_pareto = np.log(len(df_order0))*k - 2*(logLik_pareto)
        print("pareto | logLik %s AIC %s BIC %s" % (logLik_pareto, aic_pareto, bic_pareto))
        txt_out.write("pareto | logLik %s AIC %s BIC %s\n" % (logLik_pareto, aic_pareto, bic_pareto))
        if(bic_pareto < test):
            best = bic_pareto
            test = bic_pareto
            bestLABEL = "pareto"
        ### exponential function
        fit = sp.stats.expon.fit(df_order0)
        #print(fit)
        logLik_expon = np.sum(sp.stats.expon.logpdf(df_order0, fit[0], scale=fit[1]))
        k = len(fit)
        aic_expon = 2*k - 2*(logLik_expon)
        bic_expon = np.log(len(df_order0))*k - 2*(logLik_expon)
        print("expon | logLik %s AIC %s BIC %s" % (logLik_expon, aic_expon, bic_expon))
        txt_out.write("expon | logLik %s AIC %s BIC %s\n" % (logLik_expon, aic_expon, bic_expon))
        if(bic_expon < test):
            best = bic_expon
            test = bic_expon
            bestLABEL = "expon"
        ### normal function
        fit = sp.stats.norm.fit(df_order0)
        #print(fit)
        logLik_norm = np.sum(sp.stats.norm.logpdf(df_order0, fit[0], scale=fit[1]))
        k = len(fit)
        aic_norm = 2*k - 2*(logLik_norm)
        bic_norm = np.log(len(df_order0))*k - 2*(logLik_norm)
        print("norm | logLik %s AIC %s BIC %s" % (logLik_norm, aic_norm, bic_norm))
        txt_out.write("norm | logLik %s AIC %s BIC %s\n" % (logLik_norm, aic_norm, bic_norm))
        if(bic_norm < test):
            best = bic_norm
            test = bic_norm
            bestLABEL = "norm"
        ### truncated normal function
        fit = sp.stats.truncnorm.fit(df_order0)
        #print(fit)
        logLik_truncnorm = np.sum(sp.stats.truncnorm.logpdf(df_order0, fit[0], fit[1], loc=fit[2], scale=fit[3]))
        k = len(fit)
        aic_truncnorm = 2*k - 2*(logLik_truncnorm)
        bic_truncnorm = np.log(len(df_order0))*k - 2*(logLik_truncnorm)
        print("truncnorm | logLik %s AIC %s BIC %s" % (logLik_truncnorm, aic_truncnorm, bic_truncnorm))
        txt_out.write("truncnorm | logLik %s AIC %s BIC %s\n" % (logLik_truncnorm, aic_truncnorm, bic_truncnorm))
        if(bic_truncnorm < test):
            best = bic_truncnorm
            test = bic_truncnorm
            bestLABEL = "truncnorm"
        print("BEST MODEL is %s at %s" % (bestLABEL,best))
        txt_out.write("BEST MODEL is %s at %s" % (bestLABEL,best))
        txt_out.close
        
        
    
    
    
            
            
################################################################################################
################################################################################################
   
def main():
    if(fof=="file"):
        print("make histogram")    
        readPath = "%s_analysis/distances.txt" % (inp)
        df = pd.read_csv(readPath, sep = "\t")
        #print(df)
        readPath2 = "%s_analysis/distances_order1.txt" % (inp)
        df_order1 = pd.read_csv(readPath2, sep = "\t")
        readPath3 = "%s_analysis/distances_order0.txt" % (inp)
        df_order0 = pd.read_csv(readPath3, sep = "\t")
        dirname = "none"
        mm_inf(df_order0,df_order1,dirname)
        sns.histplot(df, x="distance", hue="order", kde=True)
        readPath4 = "%s_analysis/mm_inference_order1.txt" % (inp)
        df_bic = pd.read_csv(readPath4, sep = "\t", header = None)
        #print(df_bic)
        df_bic_best = df_bic[df_bic[0].str.contains('BEST')]
        df_bic_best = df_bic_best.values
        df_bic_best = str(df_bic_best[0])
        #print(df_bic_best)
        df_bic_best = df_bic_best.split(" ")
        bestLABEL_order1 = str(df_bic_best[3])
        readPath5 = "%s_analysis/mm_inference_order0.txt" % (inp)
        df_bic = pd.read_csv(readPath5, sep = "\t", header = None)
        #print(df_bic)
        df_bic_best = df_bic[df_bic[0].str.contains('BEST')]
        df_bic_best = df_bic_best.values
        df_bic_best = str(df_bic_best[0])
        #print(df_bic_best)
        df_bic_best = df_bic_best.split(" ")
        bestLABEL_order0 = str(df_bic_best[3])
        plt.title("best model %s|1st order =%s|0th order =%s" % (inp,bestLABEL_order1,bestLABEL_order0))
        plt.savefig("%s_analysis/histogram.png" % (inp))
        plt.show()
    if(fof=="folder"):
        print("make histograms")    
        lst = os.listdir("%s_analysis/intervals/" % (inp)) # your directory path
        number_files = len(lst)
        print("number of files")
        print(number_files)
        dir_list = os.listdir("%s_analysis/intervals/" % (inp))
        print(dir_list)
        for fname in dir_list:
            print(fname)
            dirname = fname
            readPath = "%s_analysis/distances_%s.txt" % (inp,dirname)
            df = pd.read_csv(readPath, sep = "\t")
            print(df)
            readPath2 = "%s_analysis/distances_order1_%s.txt" % (inp,dirname)
            df_order1 = pd.read_csv(readPath2, sep = "\t")
            readPath3 = "%s_analysis/distances_order0_%s.txt" % (inp,dirname)
            df_order0 = pd.read_csv(readPath3, sep = "\t")
            mm_inf(df_order0,df_order1,dirname)
            sns.histplot(df, x="distance", hue="order", kde=True)
            readPath4 = "%s_analysis/mm_inference_order1_%s.txt" % (inp,dirname)
            df_bic = pd.read_csv(readPath4, sep = "\t", header = None)
            #print(df_bic)
            df_bic_best = df_bic[df_bic[0].str.contains('BEST')]
            df_bic_best = df_bic_best.values
            df_bic_best = str(df_bic_best[0])
            #print(df_bic_best)
            df_bic_best = df_bic_best.split(" ")
            bestLABEL_order1 = str(df_bic_best[3])
            readPath5 = "%s_analysis/mm_inference_order0_%s.txt" % (inp,dirname)
            df_bic = pd.read_csv(readPath5, sep = "\t", header = None)
            #print(df_bic)
            df_bic_best = df_bic[df_bic[0].str.contains('BEST')]
            df_bic_best = df_bic_best.values
            df_bic_best = str(df_bic_best[0])
            #print(df_bic_best)
            df_bic_best = df_bic_best.split(" ")
            bestLABEL_order0 = str(df_bic_best[3])
            plt.title("best model %s|1st order =%s|0th order =%s" % (dirname,bestLABEL_order1,bestLABEL_order0))
            plt.savefig("%s_analysis/histogram_%s.png" % (inp,dirname))
            plt.show()
###############################################################
if __name__ == '__main__':
    main()
    
        
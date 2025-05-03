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
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Arc
import ternary
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

if(fof=="file"):
    # calculate number of faces for single file
    lst = os.listdir("%s_analysis/intervals/" % inp) # your directory path
    face_num = int(len(lst)/4)  # note folder has 4 types of files
    print("number of Chernoff faces is %s" % face_num)
    

def chernoff_face(ax, x, y, features, facecolor='lightgray', edgecolor='black'):
    """
    Draws a Chernoff face on a given matplotlib axes.

    Parameters:
    - ax: matplotlib axes object
    - x, y: coordinates of the center of the face
    - features: a list or numpy array of 11 values representing the face features
    - facecolor: color of the face
    - edgecolor: color of the face outline
    """
    #print("making Chernoff faces")
    # Normalize features to be between 0 and 1
    features = np.array(features)
    features = (features - np.min(features)) / (np.max(features) - np.min(features))

    # Face outline
    face = Ellipse((x, y), 
                   width=0.8 + features[0] * 0.01,  # Face width
                   height=1.0 + features[1] * 0.01, # Face height
                   angle=0, 
                   facecolor=facecolor, 
                   edgecolor=edgecolor)
    ax.add_patch(face)

    # Eyes
    eye_width = 0.1 + features[2] * 0.1
    eye_height = 0.1 + features[3] * 0.1
    ax.add_patch(Ellipse((x - 0.2, y + 0.2), width=eye_width, height=eye_height, facecolor='white', edgecolor='black'))
    ax.add_patch(Ellipse((x + 0.2, y + 0.2), width=eye_width, height=eye_height, facecolor='white', edgecolor='black'))

    # Pupils
    pupil_size = 0.02 + features[4] * 0.05
    ax.add_patch(Ellipse((x - 0.2, y + 0.2), width=pupil_size, height=pupil_size, facecolor='black'))
    ax.add_patch(Ellipse((x + 0.2, y + 0.2), width=pupil_size, height=pupil_size, facecolor='black'))

    # Eyebrows
    eyebrow_angle = -30 + features[5] * 60  # Angle between -30 and 30 degrees
    ax.plot([x - 0.3, x - 0.1], [y + 0.3 + 0.05 * np.sin(np.radians(eyebrow_angle)), y + 0.3 - 0.05 * np.sin(np.radians(eyebrow_angle))], color='black', linewidth=2)
    ax.plot([x + 0.1, x + 0.3], [y + 0.3 - 0.05 * np.sin(np.radians(eyebrow_angle)), y + 0.3 + 0.05 * np.sin(np.radians(eyebrow_angle))], color='black', linewidth=2)

    # Nose
    nose_width = 0.1 + features[6] * 0.01
    nose_height = 0.15 + features[7] * 0.01
    ax.add_patch(Ellipse((x, y), width=nose_width, height=nose_height, angle=features[8] * 45, facecolor='lightgray', edgecolor='black'))

    # Mouth
    mouth_width = 0.2 + features[9] * 0.1
    mouth_height = 0.05 + features[10] * 0.1
    mouth = Ellipse((x, y - 0.2), width=mouth_width, height=mouth_height, angle=0,facecolor='black', edgecolor='black')
    ax.add_patch(mouth)
    
    #Ears
    ear_width = 0.05 + features[10] * 0.1
    ear_height = 0.05 + features[10] * 0.2
    ax.add_patch(Ellipse((x - 0.5, y ), width=ear_width, height=ear_height, angle=0, facecolor=facecolor, edgecolor=edgecolor))
    ax.add_patch(Ellipse((x + 0.5, y ), width=ear_width, height=ear_height, angle=0, facecolor=facecolor, edgecolor=edgecolor))


def ternary_plot1(tdata, i, randX, randY, randZ):
    #print("making ternary plots")
    # Create a figure and axes with ternary scale
    fig, tax = ternary.figure(scale=1.0)
        
    # Plot the data points
    #print(i)
    #print(tdata)
    if(i != 0):
        current_key, current_value = list(tdata.items())[-1]
        #print(current_value)
        tax.scatter([current_value], marker='o', color='orange', label='current value')
    #tax.plot_colored_trajectory(tdata.values(), linewidth=0.8, label="trajectory")
    tax.plot_colored_trajectory(tdata.values(), linewidth=0.6, color='black', label="song trajectory")  
    max_fs = 18
    # corner font size
    fsX = randX*max_fs
    fsY = randY*max_fs
    fsZ = randZ*max_fs
    # axis font size
    fsE = (randY+randZ)*0.5*max_fs
    fsP = (randX+randY)*0.5*max_fs
    fsI = (randX+randZ)*0.5*max_fs
    # Set labels and title
    tax.right_corner_label("CONTROL", fontsize=fsX, color='black')
    tax.top_corner_label("ENERGY", fontsize=fsY, color='black')
    tax.left_corner_label("SURPRISE", fontsize=fsZ, color='black')
    tax.left_axis_label("emotional impact", fontsize=fsE, color='green') # A
    tax.right_axis_label("physical impact", fontsize=fsP, color='red') # B
    tax.bottom_axis_label("intellectual impact", fontsize=fsI, color='blue') # C
    tax.set_title("Audio Fitness Signal", fontsize=18, y=-0.15)

    # Remove default Matplotlib axes
    tax.get_axes().axis('off')

    # Add legend
    tax.legend()

    # Draw gridlines
    tax.gridlines(multiple=0.1, color="grey")
    #tax.ticks(axis='lbr', linewidth=1)
    return tax
    


def ternary_plot2(tdata, i, randX, randY, randZ):
    #print("making ternary plots")
    # Create a figure and axes with ternary scale
    fig, tax = ternary.figure(scale=1.0)
        
    # Plot the data points
    #print(i)
    #print(tdata)
    if(i != 0):
        current_key, current_value = list(tdata.items())[-1]
        #print(current_value)
        tax.scatter([current_value], marker='o', color='orange', label='current value')
    #tax.plot_colored_trajectory(tdata.values(), linewidth=0.8, label="trajectory")
    tax.plot_colored_trajectory(tdata.values(), linewidth=0.6, color='black', label="song trajectory")  
    max_fs = 18
    # corner font size
    fsX = randX*max_fs
    fsY = randY*max_fs
    fsZ = randZ*max_fs
    # axis font size
    fsE = (randY+randZ)*0.5*max_fs
    fsP = (randX+randY)*0.5*max_fs
    fsI = (randX+randZ)*0.5*max_fs
    # Set labels and title
    tax.right_corner_label("CONTROL", fontsize=fsX, color='black')
    tax.top_corner_label("ENERGY", fontsize=fsY, color='black')
    tax.left_corner_label("SURPRISE", fontsize=fsZ, color='black')
    tax.left_axis_label("emotional impact", fontsize=fsE, color='green') # A
    tax.right_axis_label("physical impact", fontsize=fsP, color='red') # B
    tax.bottom_axis_label("intellectual impact", fontsize=fsI, color='blue') # C
    tax.set_title("Lyrical Fitness Signal", fontsize=18, y=-0.15)

    # Remove default Matplotlib axes
    tax.get_axes().axis('off')

    # Add legend
    tax.legend()

    # Draw gridlines
    tax.gridlines(multiple=0.1, color="grey")
    #tax.ticks(axis='lbr', linewidth=1)
    return tax
    
    
################################################################################################
################################################################################################


def main():
    # Generate random data for faces and tplots
    np.random.seed()
    data = np.random.rand(face_num, 12)
    print(data)
       
    # import and shape external data

    ##########################    
    # Plot each Chernoff face
    ##########################
    for i in range(face_num):
        if(i>face_num-1):
            continue
        
        # Create a new figure for each subplot
        if not os.path.exists('%s_analysis/faces' % inp):
            os.mkdir('%s_analysis/faces' % inp)
        print("generating Chernoff face %s" % str(i+1))
        fig_single, ax_single = plt.subplots()
        # Copy the plot from the original subplot to the new figure
        chernoff_face(ax_single, 0, 0, data[i])
        ax_single.set_xlim(-1, 1)
        ax_single.set_ylim(-1, 1)
        ax_single.set_aspect('equal', adjustable='box')
        ax_single.axis('off')
        ax_single.set_title('Chernoff face - fitness signal', fontsize=18,)
        # Save the new figure
        fig_single.savefig(f'%s_analysis/faces/face_{i+1}.png' % inp)
        # Close the new figure to release memory
        plt.close(fig_single)
    ###########################
    # make each ternary plot 1
    ###########################
    tdata = {}
    randX = 0.5
    randY = 0.5
    randZ = 0.5  
    for i in range(face_num):
        if(i == 0):
            randX = 0.5
            randY = 0.5
            randZ = 0.5
        elif(i>0):
            # random signal
            randX = rnd.random()
            randY = rnd.random()
            randZ = rnd.random()
        tdata_name = "N%s" % i
        tdata_add = [randX, randY, randZ]
        tdata_sum = sum(tdata_add)
        # Normalize the numbers so that they sum to 1
        tdata_norm = [number / tdata_sum for number in tdata_add]
        tdata.update({tdata_name: tdata_norm})
        #tdata.append(tdata_add)
        #print(tdata)
        if(i>face_num-1):
            continue
        if not os.path.exists('%s_analysis/tplots1' % inp):
            os.mkdir('%s_analysis/tplots1' % inp)
        print("generating ternary plot 1 %s" % str(i+1))
        tax = ternary_plot1(tdata, i, randX, randY, randZ)
        # save image
        tax.savefig('%s_analysis/tplots1/tplot_%s.png' % (inp, i), dpi=144)
        tax.close()
    
    ###########################    
    # make each ternary plot 2
    ###########################
    if(lyr == "yes"):   
        tdata = {}
        randX = 0.5
        randY = 0.5
        randZ = 0.5  
        for i in range(face_num):
            if(i == 0):
                randX = 0.5
                randY = 0.5
                randZ = 0.5
            elif(i>0):
                # random signal
                randX = rnd.random()
                randY = rnd.random()
                randZ = rnd.random()
            tdata_name = "N%s" % i
            tdata_add = [randX, randY, randZ]
            tdata_sum = sum(tdata_add)
            # Normalize the numbers so that they sum to 1
            tdata_norm = [number / tdata_sum for number in tdata_add]
            tdata.update({tdata_name: tdata_norm})
            #tdata.append(tdata_add)
            #print(tdata)
            if(i>face_num-1):
                continue
            if not os.path.exists('%s_analysis/tplots2' % inp):
                os.mkdir('%s_analysis/tplots2' % inp)
            print("generating ternary plot 2 %s" % str(i+1))
            tax = ternary_plot2(tdata, i, randX, randY, randZ)
            # save image
            tax.savefig('%s_analysis/tplots2/tplot_%s.png' % (inp, i), dpi=144)
            tax.close()

##################################################################        

def create_list():   
    lst = os.listdir(inp) # your directory path
    number_files = len(lst)
    print("number of files")
    print(number_files)
    dir_list = os.listdir(inp)
    print(dir_list)
    global folder_paths1
    folder_paths1 = []
    global folder_paths2
    folder_paths2 = []
    global folder_paths3
    folder_paths3 = []
    if not os.path.exists('%s_analysis/faces' % (inp)):
        os.mkdir('%s_analysis/faces' % (inp))
    if not os.path.exists('%s_analysis/tplots1' % (inp)):
        os.mkdir('%s_analysis/tplots1' % (inp))
        if(lyr == "yes"):
            if not os.path.exists('%s_analysis/tplots2' % (inp)):
                os.mkdir('%s_analysis/tplots2' % (inp))
    for i in range(number_files):    
        # Open an mp3 file 
        filename = dir_list[i]
        dirname = filename[:-4]
        if not os.path.exists('%s_analysis/faces/%s' % (inp,dirname)):
            os.mkdir('%s_analysis/faces/%s' % (inp,dirname))
        if not os.path.exists('%s_analysis/tplots1/%s' % (inp,dirname)):
            os.mkdir('%s_analysis/tplots1/%s' % (inp,dirname))
        if(lyr == "yes"):
            if not os.path.exists('%s_analysis/tplots2/%s' % (inp,dirname)):
                os.mkdir('%s_analysis/tplots2/%s' % (inp,dirname))
        
        folder_path1 = "%s_analysis/faces/%s" % (inp,dirname)
        folder_path2 = "%s_analysis/tplots1/%s" % (inp,dirname)
        if(lyr == "yes"):
            folder_path3 = "%s_analysis/tplots2/%s" % (inp,dirname)
        #print(filename)
        print("generating faces and tplots for %s" % (dirname))
        folder_paths1.append(folder_path1)
        folder_paths2.append(folder_path2)
        if(lyr == "yes"):     
             folder_paths3.append(folder_path3)
    print(folder_paths1)
    print(folder_paths2)
    print(folder_paths3)
    return folder_paths1
    return folder_paths2
    return folder_paths3
    
#################################################################
def main_batch_faces(item):
    
    folder_path = item
    print(folder_path)
    folder_path_array = folder_path.split("/")
    filename = "%s.wav" % (folder_path_array[2])
    print(filename)
    # calculate number of faces for single file
    lst = os.listdir("%s_analysis/intervals/%s" % (inp,filename)) # your directory path
    face_num = int(len(lst)/4)  # note folder has 4 types of files
    print("number of Chernoff faces is %s" % face_num)  
    # Generate random data for faces and tplots
    np.random.seed()
    data = np.random.rand(face_num, 12)
    print(data)
    
    
    # import and shape external data

    ##########################    
    # Plot each Chernoff face
    ##########################
    for i in range(face_num):
        if(i>face_num-1):
            continue
        
        # Create a new figure for each subplot
        print("generating Chernoff face %s for %s" % (str(i+1),filename))
        fig_single, ax_single = plt.subplots()
        # Copy the plot from the original subplot to the new figure
        chernoff_face(ax_single, 0, 0, data[i])
        ax_single.set_xlim(-1, 1)
        ax_single.set_ylim(-1, 1)
        ax_single.set_aspect('equal', adjustable='box')
        ax_single.axis('off')
        ax_single.set_title('Chernoff face - fitness signal', fontsize=18,)
        # Save the new figure
        fig_single.savefig(f'%s/face_{i+1}.png' % (folder_path))
        # Close the new figure to release memory
        plt.close(fig_single)
    

def main_batch_tplots1(item):
    
    folder_path = item
    print(folder_path)
    folder_path_array = folder_path.split("/")
    filename = "%s.wav" % (folder_path_array[2])
    print(filename)
    # calculate number of faces for single file
    lst = os.listdir("%s_analysis/intervals/%s" % (inp,filename)) # your directory path
    face_num = int(len(lst)/4)  # note folder has 4 types of files
    print("number of tplots is %s" % face_num)  
    ###########################
    # make each ternary plot 1
    ###########################
    tdata = {}
    randX = 0.5
    randY = 0.5
    randZ = 0.5  
    for i in range(face_num):
        if(i == 0):
            randX = 0.5
            randY = 0.5
            randZ = 0.5
        elif(i>0):
            # random signal
            randX = rnd.random()
            randY = rnd.random()
            randZ = rnd.random()
        tdata_name = "N%s" % i
        tdata_add = [randX, randY, randZ]
        tdata_sum = sum(tdata_add)
        # Normalize the numbers so that they sum to 1
        tdata_norm = [number / tdata_sum for number in tdata_add]
        tdata.update({tdata_name: tdata_norm})
        #tdata.append(tdata_add)
        #print(tdata)
        if(i>face_num-1):
            continue
        if not os.path.exists('%s_analysis/tplots1' % inp):
            os.mkdir('%s_analysis/tplots1' % inp)
        print("generating ternary plot 1 %s for %s" % (str(i+1),filename))
        tax = ternary_plot1(tdata, i, randX, randY, randZ)
        # save image
        tax.savefig('%s/tplot_%s.png' % (folder_path, i), dpi=144)
        tax.close()
    
def main_batch_tplots2(item):
    
    folder_path = item
    print(folder_path)
    folder_path_array = folder_path.split("/")
    filename = "%s.wav" % (folder_path_array[2])
    print(filename)
    # calculate number of faces for single file
    lst = os.listdir("%s_analysis/intervals/%s" % (inp,filename)) # your directory path
    face_num = int(len(lst)/4)  # note folder has 4 types of files
    print("number of tplots is %s" % face_num)  
    ###########################
    # make each ternary plot 2
    ###########################
    tdata = {}
    randX = 0.5
    randY = 0.5
    randZ = 0.5  
    for i in range(face_num):
        if(i == 0):
            randX = 0.5
            randY = 0.5
            randZ = 0.5
        elif(i>0):
            # random signal
            randX = rnd.random()
            randY = rnd.random()
            randZ = rnd.random()
        tdata_name = "N%s" % i
        tdata_add = [randX, randY, randZ]
        tdata_sum = sum(tdata_add)
        # Normalize the numbers so that they sum to 1
        tdata_norm = [number / tdata_sum for number in tdata_add]
        tdata.update({tdata_name: tdata_norm})
        #tdata.append(tdata_add)
        #print(tdata)
        if(i>face_num-1):
            continue
        if not os.path.exists('%s_analysis/tplots2' % inp):
            os.mkdir('%s_analysis/tplots2' % inp)
        print("generating ternary plot 2 %s for %s" % (str(i+1),filename))
        ternary_plot1(tdata, i, randX, randY, randZ)
        # save image
        tax.savefig('%s/tplot_%s.png' % (folder_path, i), dpi=144)
        tax.close()   


###############################################################
if __name__ == '__main__':
    if(fof == "file"):
        main()
    if(fof == "folder"):
        create_list()
        with multiprocessing.Pool(processes=num_cores) as pool: # Use os.cpu_count() for max processes
            pool.map(main_batch_faces, folder_paths1)
        with multiprocessing.Pool(processes=num_cores) as pool: # Use os.cpu_count() for max processes
            pool.map(main_batch_tplots1, folder_paths2)
        if(lyr == "yes"):
            with multiprocessing.Pool(processes=num_cores) as pool: # Use os.cpu_count() for max processes
                pool.map(main_batch_tplots2, folder_paths3)
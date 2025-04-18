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
 ###### variable assignments ######
inp = ""+name+""
tm = int(tm)

# calculate number of faces for single file
lst = os.listdir("%s_analysis/intervals/" % inp) # your directory path
face_num = int(len(lst)/4)  # note folder has 4 types of files
print("number of Chernoff faces is %s" % face_num)
# calculate number of faces for folder

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

def main():
    # Generate random data for 10 faces
    np.random.seed()
    data = np.random.rand(face_num, 12)
    print(data)
    # import and shape external data

    # Create the plot
    fig, axes = plt.subplots(int(face_num/5), int(face_num/2), figsize=(15, 6))
    axes = axes.flatten()
    #print(axes)
    # Plot each Chernoff face
    for i, ax in enumerate(axes):
        if(i>face_num-1):
            continue
        '''
        # create multipanel face plot
        chernoff_face(ax, 0, 0, data[i])
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_aspect('equal', adjustable='box')
        ax.axis('off')
        ax.set_title('fitness signal - Chernoff face')
        '''
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
        ax_single.set_title('fitness signal - Chernoff face')
        # Save the new figure
        fig_single.savefig(f'%s_analysis/faces/face_{i+1}.png' % inp)
        # Close the new figure to release memory
        plt.close(fig_single)
    
    '''   
    # finish multipanel face plot
    plt.tight_layout()
    plt.savefig('chernoff.png')
    plt.show()
    '''
    

###############################################################
if __name__ == '__main__':
    main()
    
    
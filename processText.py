#!/usr/bin/env python

#############################################################################
######   POPSTAR software for detecting fitness signaling in music
######   produced by Dr. Gregory A. Babbitt
######   and students at the Rochester Instituteof Technology in 2025.
######   Offered freely without guarantee.  License under GPL v3.0
#############################################################################


import getopt, sys # Allows for command line arguments
import os
import gc
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import math
import random as rnd
import re
import time
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, BitsAndBytesConfig
from sklearn.decomposition import PCA
import multiprocessing
import textwrap
from PIL import Image, ImageDraw, ImageFont
# IMPORTANT NOTE - run in base conda env, not in atomdance conda env   
################################################################################

print("\nIT APPEARS YOU ARE ANALYZING TEXT AND NOT AUDIOVISUAL FORMAT...")
#tm = input("\nEnter the text chunk length to analyze (in tokens) (e.g. 600)\n")
#stl = input("\nEnter the text chunk window step length (in tokens) (e.g. 300)\n")

tm = 500
stl = 250


inp0 = "full"
inp00 = "y"
#inp = "test"
#tm = 20 # interval length in seconds

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
    #if(header == "int"):
    #    tm = value
    #    print("my time interval is",tm)
    if(header == "metro"):
        met = value
        print("my metronome option is",met)
    if(header == "spch"):
        spchOpt = value
        print("speech-normalize",spchOpt)
    if(header == "self"):
        selfOpt = value
        print("self-normalize",selfOpt)
    if(header == "musi"):
        musiOpt = value
        print("music-normalize",musiOpt)
    if(header == "fileExt"):
        ext = value
        print("my file extension(s) are",ext)   
infile.close()
 ###### variable assignments ######
inp = ""+name+""
tm = int(tm)
stl = int(stl)
met = ""+met+""
spchOpt = ""+spchOpt+""
selfOpt = ""+selfOpt+""
musiOpt = ""+musiOpt+""
ext = ""+ext+""

'''
if os.path.exists('%s_analysis' % inp):
    print("folder already exists...was this run already done?")
    exit()
'''
if not os.path.exists('%s_analysis' % inp):
        os.mkdir('%s_analysis' % inp)
if not os.path.exists('%s_analysis/texts' % inp):
        os.mkdir('%s_analysis/texts' % inp)        

################################################################################
#########################   sonogram generator  ################################
################################################################################
#infile= input("\nEnter path/name of input sound file (name.wav)\n")   
#outfile = input("\nEnter path/name of output image file (name.png)\n")
# IMPORTANT NOTE - run in base conda env, not in atomdance conda env  
input0 = "%s.wav" % inp
input1 = "%s.mp4" % inp
input2 = "%s.png" % inp
input3 = "%s.dat" % inp
input4 = "%s.txt" % inp
input5 = "%s.jpg" % inp

if os.path.isfile(input4):
    print("user input is a .txt file")
    fileORfolder = "file"
    #inp2 = input("Do you want to activate bootstrapping? (y or n)\n")
elif os.path.isdir(inp):
    print("user input is a folder")
    fileORfolder = "folder"
else:
    print("Invalid Path")
    exit()
    
#################################################################################
####################  preprocessing     #########################################
#################################################################################

 
def cos_sim():   
    # Initialize a lightweight LLM model and tokenizer
    #model_name = input("\nEnter LLM model (e.g. sentence-transformers/all-MiniLM-L6-v2)\n")
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    '''
    model_name = "meta-llama/Meta-Llama-3-8B"
    hf_token = input("\nEnter Hugging Face token\n")
    hf_token = "YOUR_HUGGINGFACE_TOKEN_HERE" # Replace with your token
    quantization_config = BitsAndBytesConfig(load_in_8bit=True,) # Compresses model footprint down to ~9GB VRAM
    print("Loading Llama 3 (8B) into memory...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        quantization_config=quantization_config,
        device_map="auto",
        token=hf_token
    )
    '''
    # Tokenize some text
    #sentence = "go go go go go go go now"
    #sentence = "The artificial intelligence model extracted deep contextual embeddings."
    #sentence = "The financial bank is closed today, but the muddy river bank is still accessible."
    #sentence = "When forty winters shall besiege thy brow, And dig deep trenches in thy beauty's field, Thy youth's proud livery so gazed on now,  Will be a tattered weed of small worth held: Then being asked, where all thy beauty lies, Where all the treasure of thy lusty days."
    if(fileORfolder == "file"):
        file_path = input4  # Replace with your actual file path
    if(fileORfolder == "folder"):
        file_path = inp  # Replace with your actual file path
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            full_text = file.read()
        print(f"Successfully read {len(full_text)} characters from '{file_path}'.")
    except FileNotFoundError:
         print(f"Error: The file '{file_path}' was not found.")
         exit()
    #print(full_text)
    #tm = 600
    win_length = tm
    step_length = stl
    text_length = len(full_text)
    n_steps = int((text_length/step_length))
    print("n_steps = %s" % n_steps)
    start = 0
    end = start + step_length
    outfile = open("%s_analysis/cos_sim_%s.txt" % (inp,inp), "w")
    outfile.write("raw\tcorrect\n")
    for i in range(n_steps):
        text_chunk = full_text[start:end]
        print("win_start = %s | win_end = %s | full_text = %s" % (start, end, text_length))
        #print(text_chunk)
        # Tokenize text chunk
        inputs = tokenizer(text_chunk, return_tensors="pt")
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        # Extract raw token embeddings from the last hidden state
        with torch.no_grad():
            outputs = model(**inputs)
            # Shape: [sequence_length, hidden_dimension]
            raw_embeddings = outputs.last_hidden_state.squeeze(0).numpy()

        # Correct for Anisotropy via Mean-Centering
        sequence_mean = np.mean(raw_embeddings, axis=0, keepdims=True)
        corrected_embeddings = raw_embeddings - sequence_mean

        # Compute Pairwise Cosine Similarities
        raw_similarity = compute_pairwise_cosine_similarity(raw_embeddings)
        corrected_similarity = compute_pairwise_cosine_similarity(corrected_embeddings)

        # Calculate Average Cosine Similarity across unique word pairs
        # k=1 excludes the diagonal self-similarity (1.0)
        row_indices, col_indices = np.triu_indices(len(tokens), k=1)

        avg_raw_similarity = np.mean(raw_similarity[row_indices, col_indices])
        avg_corrected_similarity = np.mean(corrected_similarity[row_indices, col_indices])

        # Print Results
        print(f"text chunk: '{text_chunk}'\n")
        print(f"{'METRIC':<35} | {'VALUE':<15}")
        print("-" * 55)
        print(f"{'Average RAW Pairwise Similarity':<35} | {avg_raw_similarity:.4f}")
        print(f"{'Average CORRECTED Pairwise Similarity':<35} | {avg_corrected_similarity:.4f}\n")
        outfile.write("%s\t%s\n" % (avg_raw_similarity,avg_corrected_similarity))
        # save text chunk to image
        text_to_png(text_chunk, "%s_analysis/texts/text_%s.png" % (inp,i))
        start = start + step_length
        end = start + win_length
    outfile.close()    
        
def compute_pairwise_cosine_similarity(embeddings):
    dot_product = np.dot(embeddings, embeddings.T)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms_matrix = np.dot(norms, norms.T)
    norms_matrix[norms_matrix == 0] = 1e-9
    return dot_product / norms_matrix

def derive():
    print("differentiating cosine similarity")
    readpath = "%s_analysis/cos_sim_%s.txt" % (inp,inp)
    df = pd.read_csv(readpath, delimiter = "\t")
    print(df)
    control = df['correct'].to_numpy()
    energy = np.diff(control)
    surprise = np.diff(energy)
    control = control[2:] # remove first 2 elemnets of array
    energy = energy[1:] # remove first element of array
    c_len = len(control)
    e_len = len(energy)
    s_len = len(surprise)
    print("arr len %s %s %s" % (c_len, e_len, s_len))
    # Combine into a DataFrame
    ternary_raw = pd.DataFrame({
        'energy': energy,
        'control': control,
        'surprise': surprise
    })
    print(ternary_raw)
    writepath1 = "%s_analysis/ternary_txt_raw.txt" % (inp)
    ternary_raw.to_csv(writepath1, index=False)
    ternary_norm_prelim = (ternary_raw - ternary_raw.min()) / (ternary_raw.max() - ternary_raw.min())
    writepath2 = "%s_analysis/ternary_txt_norm_prelim.txt" % (inp)
    ternary_norm_prelim.to_csv(writepath2, index=False)
    ternary_norm = ternary_norm_prelim.div(ternary_norm_prelim.sum(axis=1), axis=0)
    writepath3 = "%s_analysis/ternary_txt_norm.txt" % (inp)
    ternary_norm.to_csv(writepath3, index=False)
    
def text_to_png(text, output_path="%s_analysis/texts/output.png" % (inp), wrap_width=60, padding=30):
    # 1. Wrap long paragraphs into clean lines
    wrapped_lines = []
    for paragraph in text.splitlines():
        if paragraph.strip() == "":
            wrapped_lines.append("")
        else:
            wrapped_lines.extend(textwrap.wrap(paragraph, width=wrap_width))
    formatted_text = "\n".join(wrapped_lines)

    # 2. Load a font (falls back to default if custom TTF isn't found)
    try:
      # Replace with a path to a .ttf file on your system (e.g., "arial.ttf" or "/System/Library/Fonts/Menlo.ttc")
      font = ImageFont.truetype("arial.ttf", size=20)
    except IOError:
      font = ImageFont.load_default()

    # 3. Calculate exact text dimensions using a temporary dummy image
    dummy_img = Image.new("RGB", (1, 1))
    dummy_draw = ImageDraw.Draw(dummy_img)
    bbox = dummy_draw.multiline_textbbox(
        (0, 0), formatted_text, font=font, spacing=6
    )

    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # 4. Create the final canvas with padding and draw the text
    img_width = text_width + (padding * 2)
    img_height = text_height + (padding * 2)

    img = Image.new("RGB", (img_width, img_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.multiline_text(
      (padding, padding),
      formatted_text,
      fill=(20, 20, 20),
      font=font,
      spacing=6,
    )

    # 5. Save as PNG
    img.save(output_path, "PNG")
    print(f"Saved image to {output_path}")  
    
#################################################################################
####################  main program      #########################################
#################################################################################
def main():
    if(fileORfolder == "file"):
        # Path to your TXT file
        fpath = "%s" % input4
        file_size_bytes = os.path.getsize(fpath) # Get size in bytes
        file_size_mb = file_size_bytes / (1024 * 1024)# Convert to Megabytes (MB)
        print(f"File Size: {file_size_mb:.2f} MB")
        time.sleep(1)
        print("calculating control = avg corrected cosine similarity")
        time.sleep(1)
        cos_sim()
        print("calculating energy and surprise = 1st/2nd derivatives of cosine similarity")
        time.sleep(1)
        derive()
        
        print("\ntext processing is complete\n")
        
    if(fileORfolder == "folder"):
        
        print("\ntext processing is complete\n")
###############################################################
if __name__ == '__main__':
    main()
    
    
# POPSTAR-signaling-in-music

This repo is for code-base and PyQt5 applications for investigating fitness signaling in human/animal vocal contexts.  Sound files are segmented, analyzed and coverted to dynamic Chernoff faces and ternary plot trjectories represented as movies.

As of 2025, this project is in active development by undergraduate students at the Rochester Institute of Technology working with Dr. Gregory A Babbitt (Life Sciences) and Dr. Ernest P Fokoue (mathematics). Community collaboration is welcome...please reach out to us via email at RIT with an inquiry for more details (gabsbi(at)rit.edu or epfeqa(at)rit.edu).

TESTING: Linux Mint 20 (probably works on other Debian Linux distros and MacOS too).  It is not yet tested on Windows (can use a Linux Mint VM from virtualBox for now)

TO RUN: python3 popstar.py 

INPUT: myFile.wav or myFile.mp3 or myDir (...where myDir is a directory with multiple sound files for batch processing) Files provided for testing the code are test.mp3 and the folder test-cases which has 3 copies of this file. 

DEPENDENCIES: PyQt5, PIL, python-ternary, hurst, EntropyHub, FactorAnalyzer, librosa, soundfile, pydub, moviepy (use pip install moviepy==1.0.3 to avoid recent bug), and the usual suspects for data analysis (numpy, scipy, pandas, matplotlib, statsmodels, multiprocessing)

Quick look at the GUI

https://github.com/gbabbitt/POPSTAR-signaling-in-music/blob/main/POPSTARgui.png




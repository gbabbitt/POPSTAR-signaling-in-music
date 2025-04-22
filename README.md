# POPSTAR-signaling-in-music

This repo is for code-base and PyQt5 applications for investigating fitness signaling in human/animal vocal contexts.  Sound files are segmented, analyzed and coverted to dynamic Chernoff faces and ternary plot trjectories represented as movies.

As of 2025, this project is in active development by undergraduate students at the Rochester Institute of Technology working with Dr. Gregory A Babbitt (Life Sciences) and Dr. Ernest P Fokoue (mathematics). Community collaboration is welcome...please reach out to us via email at RIT with an inquiry for more details (gabsbi(at)rit.edu or epfeqa(at)rit.edu).

TESTING: Linux Mint 20 (probably works on other Debian Linux distros and MacOS too).  It is not yet tested on Windows (can use a Linux Mint VM from virtualBox for now)

TO RUN: python3 popstar.py 

INPUT: myFile.wav or myFile.mp3

DEPENDENCIES: PyQt5, PIL, python-ternary, soundfile, pydub, moviepy, and the usual suspects (numpy,scipy,pandas, matplotlib, statsmodels)

NOTE: This code does not actually do any significant analysis as of yet. The signals plotted dynamically are just random for now...stay tuned



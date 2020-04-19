from __future__ import division
from __future__ import print_function

import sys
import math
import pickle
import copy
import numpy as np
import cv2
import matplotlib.pyplot as plt
from src.DataLoader import Batch
from src.SamplePreprocessor import preprocess

# Define constants like file paths

class Constants:
    "filenames and paths to data"
    fnCharList = '../model/charList.txt'
    fnAnalyze = '../data/analyze.png'
    fnPixelRelevance = '../data/pixelRelevance.npy'
    fnTranslationInvariance = '../data/translationInvariance.npy'
    fnTranslationInvarianceTexts = '../data/translationInvarianceTexts.pickle'
    gtText = 'are'
    distribution = 'histogram' # 'histogram' or 'uniform'

def odds(val):
    return val/(1-val)

def weightOfEvidence(origProb,margProb):
    return math.log2(odds(origProb)) - math.log2(odds(margProb))

def analyzePixelRelevance():
    "simplified implementation of paper: Zintgraf et al - Visualizing Deep Neural Network Decisions: Prediction Difference"

    #setup model
    pass
from __future__ import division
from __future__ import print_function

import sys
import argparse
import cv2
import editdistance
from src.DataLoader import DataLoader, Batch
from src.Model import Model, DecoderType
from src.SamplePreprocessor import preprocess


class FilePaths:
    "filenames and paths to data"
    fnCharList = '../model/charList.txt'
    fnAccuracy = '../model/accuracy.txt'
    fnTrain = '../data/'
    fnInfer = '../data/test.png'
    fnCorpus = '../data/corpus.txt'


def train(model, loader):
    "train NN"
    epohs = 0  # number of training epochs since start
    bestCharErrorRate = float('inf') #best validation character error rate

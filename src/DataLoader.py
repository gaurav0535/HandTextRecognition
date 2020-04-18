from __future__ import division
from __future__ import print_function

import os
import random
import numpy as np
import cv2
from src.SamplePreprocessor import preprocess


class Sample:
    "pick sample frmo the dataset"

    def __init__(self, gText, filepath):
        self.gText = gText
        self.filepath = filepath


class Batch:
    "batch containing images and groud truth texts"

    def __init__(self, gText, imgs):
        self.imgs = np.stack(imgs, axis=0)
        self.gText = gText


class DataLoader:
    "loads data according to IAM format"

    def __init__(self, filePath, batchSize, imgSize, maxTextLen):
        "Loader for dataset at given location ,preprocess images and text according to parameters "

        assert filePath[-1] == '/'
        self.dataAugmentation = False
        self.currIDx = 0
        self.batchSize = batchSize
        self.imgSize = imgSize
        self.samples = []

        f = open(filePath + 'words.txt')
        chars = set()
        bad_samples = []
        bad_samples_reference = ['a01-117-05-02.png', 'r06-022-03-05.png']
        for line in f:
            # ignore comment lined they use to start with #
            if not line or line[0] == '#':
                continue

            lineSplit = line.strip().split(' ')
            assert len(lineSplit) >= 9

            # filename: part1-part2-part3 --> part1/part1-part2/part1-part2-part3.png
            fileNameSplit = lineSplit[0].split('-')
            fileName = filePath + 'words/' + fileNameSplit[0] + '/' + fileNameSplit[0] + '-' + fileNameSplit[1] + '/' + \
                       lineSplit[0] + '.png'

            # GT text are vcolumn starting at 9
            gtText = self.truncateLabel(' '.join(lineSplit[8:]), maxTextLen)
            chars = chars.union(set(list(gtText)))

            # Check if image is not empty
            if not os.path.getsize(fileName):
                bad_samples.append(lineSplit[0] + '.png')
                continue

            # put samples into list
            self.samples.append((Sample(gtText, fileName)))

        # Some images in the IAM dataset are known to be damaged , don't show warnings for them
        if set(bad_samples) != set(bad_samples_reference):
            print("Warning ,damaged images found:",bad_samples)
            print("Damaged images expected :",bad_samples_reference)

        # split into training and validation set 95% - 5%
        splitIdx = int(.95*len(self.samples))
        self.trainSamples = self.samples[:splitIdx]
        self.validationSamples = self.samples[splitIdx:]

        # put words into lists
        self.trainWords = [x.gtText for x in self.trainSamples]
        self.validationWords = [x.gtText for x in self.validationSamples]

        # number of randomly chosen samples per epoch for training
        self.numTrainSamplesPerEpoch = 25000

        # start with train set
        self.trainSet()

        # list of all chars in dataset
        self.charList = sorted(list(chars))

    def truncateLabel(self,text,maxTextLen):

        # ctc_loss cannot compute loss if it cannot find a mapping between the text label and input label
        # labels.Repeat letters cost double of the blank symbol needing to be inserted .
        # If a too-long label is provided ,  ctc_loss returns infinite gradient

        cost = 0
        for i in range(len(text)):
            if i != 0 and text[i] == text[i-1]:
                cost += 2
            else:
                cost += 1

            if cost > maxTextLen:
                return  text[:i]

        return text

    def trainSet(self):
        "Switch to randomly chosen subset of training set"

        self.dataAugmentation = True
        self.currIDx = 0
        random.shuffle(self.trainSamples)
        self.samples = self.trainSamples[:self.numTrainSamplesPerEpoch]

    def validationSet(self):
        "Switch to validation switch"

        self.dataAugmentation = False
        self.currIDx = 0
        self.samples = self.validationSamples

    def getIteratorInfo(self):
        "current batch index and overall number of batches "

        return (self.currIdx // self.batchSize + 1, len(self.samples) // self.batchSize)

    def hasNext(self):
        "itarator"

        return  self.currIDx + self.batchSize <= len(self.samples)

    def getNext(self):
        "iterator"

        batchRange = range(self.currIDx,self.currIDx + self.batchSize)
        gtTexts = [self.samples[i].gText for i in batchRange]
        imgs = [
            preprocess(cv2.imread(self.samples[i].filePath, cv2.IMREAD_GRAYSCALE), self.imgSize, self.dataAugmentation)
            for i in batchRange]
        self.currIDx += self.batchSize

        return Batch(gtTexts,imgs)





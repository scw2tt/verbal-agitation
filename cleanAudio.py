"""
Sean Wolfe
Filename: cleanAudio.py
"""

import numpy as np
from scipy.io.wavfile import write as wavWrite
from scipy.io.wavfile import read as wavRead
import matplotlib.pyplot as plt

# constants
MIC_FREQ = 10000

def cleanADC(filename):
    # find out how many lines are in the file
    count = len(open(filename).readlines(  ))
    
    # file that will be filled with audio data
    cleanAudio = []
    avgAudio = []
    door1 = []
    door2 = []
    time = []
    audioTime = []

    # open the file to read line by line and extract the ADC audio
    with open(filename, "r") as rawADC:
        # this variable holds all of the ADC in the file
        rawADC_string = rawADC.readlines()

        # get the number of lines
        rawLength = len(rawADC_string)
        
        # chop off the first few lines, and start the loop
        for i in range (3, rawLength):
            line = rawADC_string[i]
            lineList = line.split(',') # separates all values by comma
            lineLen = len(lineList)
            
            # takes care of cases with not a lot of data
            if (lineLen > 22):
                audioList = lineList[1:lineLen - 22]
                # take this list and average it
                

                audioSum = 0
                for point in audioList:
                    value = float(point)
                    audioSum += value
                    """
                    if (value < 0):
                        value = value * -1
                    """
                    cleanAudio.append(value)
                
                # compute the average of the audio and append it
                avg = audioSum/len(audioList) 
                avgAudio.append(avg)

                # averages the ten door samples and appends them
                d1List = lineList[lineLen -21: lineLen -11]
                d1 = []
                for i in d1List:
                    j = float(i)
                    d1.append(j)

                door1.append((sum(d1))/(len(d1)))

                d2List = lineList[lineLen - 11: lineLen - 1]
                d2 = []
                for i in d2List:
                    j = float(i)
                    d2.append(j)

                door2.append((sum(d2))/(len(d2)))

                
                # append the time stamp to get our t - axis
                time.append(lineList[0])

                # space out the vector so that the raw audio can be better seen
                tstamp = lineList[0]
                for i in range(len(audioList)):
                    audioTime.append(float(tstamp) + i*(1.1/len(audioList)))

        # close the file   
        rawADC.close()
    return audioTime, cleanAudio, avgAudio, time, d1, d2


# converts the list into a wav file
def listToWav(rawList, filename):
    """converts a python list of audio time and audio into a wav file
    
    Arguments:
        rawList {[float]} -- 2 column list of time and audio
        filename {string} -- name of the file to which the wav file will output
    """

    # convert the list into a numpy array
    rawArray = np.asarray(rawList)

    # write the array into the wav file
    wavWrite(filename, MIC_FREQ, rawArray)
    
    return


# takes a wav file and removes the "clicking" peaks caused by inaccuracies
def filterWav(filename, n, outfileName):
    """removes the "clicking" of the original wave file by median filtering
    
    Arguments:
        filename {string} -- input wav file
        n {int} -- median filter window length
        outfileName {string} -- the name of the wav file to be created by the function
    """


    filteredArray = []
    # convert the wav file back to a numpy array
    rate, roughAudio = wavRead(filename)

    # loop through the list to smooth the data
    remainder = roughAudio.size % n
    for i in range(0,roughAudio.size - remainder):
        # window that the median will be taken from
        window = roughAudio[i: i + n]
        # place the median in the point
        windowMed = np.median(window)

        # append the value to the output list
        filteredArray.append(windowMed)
        #if (i % 150 == 0):
            #print(windowMed)
    
    # use the remainder to fill in the remaining array to prevent falling out of bounds
    
    listToWav(filteredArray, outfileName)

    return
        

# takes in a list and makes a wav file out of it
def filterList(inList, n, outfileName):
    """converts a list of raw audio data from the ADC and creates a filtered wav file
    
    Arguments:
        inList {[int]} -- audio signal in a python list 
        n {int} -- length of median window for filtering
        outfileName {string} -- name of the wav file created by this function
    """

    # convert the list to a dummy wav that will be turned into the file they want to 
    listToWav(inList, "temp.wav")

    # take that choppy wav file and change it to a different one
    filterWav("temp.wav", n, outfileName)
    
    return


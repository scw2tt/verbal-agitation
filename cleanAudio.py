"""
Sean Wolfe
Filename: cleanAudio.py


CleanADC
Purpose: reads the text file data into python lists that can be used for analysis 
Input: filename: the text file that contains all the data readings
Returns: the following python lists
time, audioTime, cleanAudio, avgAudio, door1, door2
    1) time: the time data
    2) audioTime: the audio time (smoothed inbetween)
    3) cleanAudio: the normalized ADC readings
    4) avgAudio: the normalized ADC readings that are averaged over the window size
    5) door1: the data from the door sensor 1 (top)
    6) door2: the data from the second door sensor (bottom)

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
    return audioTime, cleanAudio


# converts the list into a wav file
def listToWav(rawList, filename):

    # convert the list into a numpy array
    rawArray = np.asarray(rawList)

    # write the array into the wav file
    wavWrite(filename, MIC_FREQ, rawArray)
    
    return


# takes a wav file and removes the "clicking" peaks caused by inaccuracies
def filterWav(filename, n, outfileName):

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
    """
    lastWindow = roughAudio.size - remainder - 1
    for i in range(remainder):
        filteredArray.append(roughAudio[lastWindow + i])

    # convert the list back to the wav format
    """
    listToWav(filteredArray, outfileName)

    return
        

# takes in a list and makes a wav file out of it
def filterList(inList, n, outfileName):

    # convert the list to a dummy wav that will be turned into the file they want to 
    listToWav(inList, "temp.wav")

    # take that choppy wav file and change it to a different one
    filterWav("temp.wav", n, outfileName)
    
    return





# test the functions
#audioFrame = cleanADC("./labData/labAgitation/rawADC2018-03-12_10-18-39.txt")
#listToWav(audioFrame, "raw.wav")

# get all of the data
"""
time, aTime,audio, meanAudio, s1, s2 = cleanADC("rawADC2018-04-21_12-35-43.txt")

plt.subplot(211)
#plt.plot(time, meanAudio, time , s1, time , s2)
plt.plot(time[300:], s1[300:],  label='Door 1') 
plt.plot(time[300:], s2[300:],  label='Door 2') 
#plt.plot(aTime, audio, label='Audio')
plt.legend()
plt.xlabel('Time (Seconds)')
plt.ylabel('Sensor Magnitude')
plt.grid(True)
plt.title('Door Sensor Radial Vision Test')

plt.subplot(212)
plt.plot(aTime, audio)
plt.grid(True)
plt.show()
#plt.legend((meanAudio, s1, s2), ('Audio', 'Door Sensor 1', 'Door Sensor 1'))
#plt.show()
"""

from pyAudioAnalysis import audioBasicIO as aIO
from pyAudioAnalysis import audioSegmentation as aS
import cleanAudio as clAudio
import matplotlib.pyplot as plt
import pydub

pydub.AudioSegment.converter = r"C:\\Python27\\ffmpeg-4.0-win64-static\\bin\\ffmpeg.exe"


# this file removes the silence from the data set using dynamic thresholding
# the function returns a set of segments where there are sound events


clAudio.filterWav("raw.wav", 400, "rawSmooth1.wav")

[Fs, x] = aIO.readAudioFile("rawSmooth1.wav")
segments = aS.silenceRemoval(x, Fs, 0.030, 0.015, smoothWindow = 0.5, Weight = 0.5, plot = True)

# plot the actual audio to compare

aTime, audio  = clAudio.cleanADC("rawADC2018-03-12_10-18-39.txt")
plt.plot(aTime, audio)
plt.show()

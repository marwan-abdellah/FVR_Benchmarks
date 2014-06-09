#!/usr/bin/python 

from __future__ import division 
import numpy as np
import matplotlib.pyplot as plt
from random import random 
import math
import sys 

print ("Profiling")

_debug_ = False

if (len(sys.argv) < 3):
	print ("Please enter the experiment prefix, \
						for instance <BENCH_PREFIX> <NUM_ITERATIONS> ")
	exit(0)

# Get the data from the command line 
filePrefix = sys.argv[1]
numIterations = sys.argv[2]

# Size range for the experiments 
sizeRange = []
if ("2D" in filePrefix):
	sizeRange = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
else:
	sizeRange = [4, 8, 16, 32, 64, 128, 256, 512]

if _debug_:
	print ("The results will be printed for " 
								+ str(len(sizeRange)) + " size(_Sec)")

# Average and SD for the entire experiment  
avgProfiles = [[0 for i in range(3)] for i in range(len(sizeRange))]
sdProfiles = [[0 for i in range(3)] for i in range(len(sizeRange))]

# Iteration of the size, to be able to index the array 
sizeItr = 0
for dataSize in sizeRange:
	
	if _debug_:
		print ("Array size is " + str(sizeRange[sizeItr]))
		
	# File full title 
	fileTitle = ""
	if ("2D" in filePrefix):
		fileTitle= filePrefix + "__%dx%d__" % (dataSize, dataSize)
	else:
		fileTitle= filePrefix + "__%dx%dx%d__" \
									% (dataSize, dataSize, dataSize)

	# Benchmarks range 
	itrRange = range(0, int(numIterations))

	# Average values for each time resolution 
	avgMicroSec 	= 0
	avgMilliSec 	= 0
	avgSec 	= 0

	# The measured data from the files
	itrProfiles = [[0 for i in range(3)] for i in itrRange]
	itrAvgProfiles = [[0 for i in range(3)]]
	itrSDProfiles = [[0 for i in range(3)] for i in itrRange]
	itrMaxSDProfiles = [[0 for i in range(3)] for i in range(1)]

	# Iterate on each file and bring the data and also calculate the average 
	for i in itrRange:
		fileName = fileTitle + str(i)
		filePtr = open(fileName, 'r') 
		
		for line in filePtr:
			if _debug_:
				print line
			
			# Parse the string to get the data 
			data = line.split(" ")
			if _debug_:
				print data
			
			# data[0] = Function name 
			# data[1] = _MicroSec
			# data[2] = _MilliSec
			# data[3] = _Sec
			
			_MicroSec = data[1]
			_MicroSec = _MicroSec.replace("us", "")
			_MicroSec = _MicroSec.replace("[", "")
			_MicroSec = _MicroSec.replace("]", "")
			_MicroSec = float(_MicroSec)
			
			_MilliSec = data[2]
			_MilliSec = _MilliSec.replace("ms", "")
			_MilliSec = _MilliSec.replace("[", "")
			_MilliSec = _MilliSec.replace("]", "")
			_MilliSec = float(_MilliSec)
			
			_Sec  = data[3]
			_Sec = _Sec.replace("s", "")
			_Sec = _Sec.replace("[", "")
			_Sec = _Sec.replace("]", "")
			_Sec = float(_Sec)
			
			# Use the seconds if their value is greater than 1 
			if (_MilliSec < 1 or _Sec > 1):
				_MilliSec = _Sec * 1000
				
			itrProfiles[i][0] = _MicroSec
			itrProfiles[i][1] = _MilliSec
			itrProfiles[i][2] = _Sec
			
			if _debug_:
				print (str(_MicroSec) + " " + str(_MilliSec) + " " + str(_Sec))
			
			# Summing the data, and later divide them by N 
			avgMicroSec 	+= _MilliSec * 1000 
			avgMilliSec 	+= _MilliSec 
			avgSec 			+= _MilliSec / 1000

	itrAvgProfiles[0][0] = avgMicroSec / len(itrRange)
	itrAvgProfiles[0][1] = avgMilliSec / len(itrRange)
	itrAvgProfiles[0][2] = avgSec 	 / len(itrRange)

	if _debug_:
		print ("avgMicroSec = %f, avgMilliSec = %f, avgSec = %f ") % \
			(itrAvgProfiles[0][0], itrAvgProfiles[0][1], itrAvgProfiles[0][2])	
			

	# Compute the SD (upper and lower bound) 
	# Consider the maximum variance, and use it for the upper and lower bounds 
	for i in itrRange:
		itrSDProfiles[i][0] = math.sqrt(math.pow(itrProfiles[i][0] - 
											itrAvgProfiles[0][0], 2))
		itrSDProfiles[i][1] = math.sqrt(math.pow(itrProfiles[i][1] - 
											itrAvgProfiles[0][1], 2))
		itrSDProfiles[i][2] = math.sqrt(math.pow(itrProfiles[i][2] - 
											itrAvgProfiles[0][2], 2)) 
		
	maxSDValue = 0
	for i in itrRange:
		if (itrSDProfiles[i][1] > maxSDValue):
			maxSDValue = itrSDProfiles[i][1]; 
			itrMaxSDProfiles[0][0] = itrSDProfiles[i][0]
			itrMaxSDProfiles[0][1] = itrSDProfiles[i][1]
			itrMaxSDProfiles[0][2] = itrSDProfiles[i][2]

	if _debug_:
		print ("sd_us = %f, sd_ms = %f, sd_s = %f ") % \
			(itrMaxSDProfiles[0][0], 
			 itrMaxSDProfiles[0][1], 
			 itrMaxSDProfiles[0][2])	
	
	if _debug_:
		print itrMaxSDProfiles
	
	# Add the data to the final array 
	avgProfiles[sizeItr][0] = itrAvgProfiles[0][0]
	avgProfiles[sizeItr][1] = itrAvgProfiles[0][1]
	avgProfiles[sizeItr][2] = itrAvgProfiles[0][2]
	
	sdProfiles[sizeItr][0] = itrMaxSDProfiles[0][0]
	sdProfiles[sizeItr][1] = itrMaxSDProfiles[0][1]
	sdProfiles[sizeItr][2] = itrMaxSDProfiles[0][2]
	
	sizeItr += 1

# Fetch the _MilliSec Data only 
avgProfileMilliSec = [0 for i in range(len(sizeRange))]
sdProfileMilliSec = [0 for i in range(len(sizeRange))]

iSize_itr = 0
for iSize in sizeRange:
	avgProfileMilliSec[iSize_itr] = avgProfiles[iSize_itr][1]
	sdProfileMilliSec[iSize_itr] = sdProfiles[iSize_itr][1]
	iSize_itr += 1

print ("avgProfileMilliSec")
print (avgProfileMilliSec)

print ("sdProfileMilliSec")
print (sdProfileMilliSec)


# Generate the data to a single file for nice plotting 
filePrefix_ = filePrefix.replace("Build/", "")
filePlot = open(filePrefix_, 'w')
for i in range(len(sizeRange)):
	filePlot.write(str(sizeRange[i]) + " ")
filePlot.write("\n")
for i in range(len(sizeRange)):
	filePlot.write(str(avgProfileMilliSec[i]) + " ")
filePlot.write("\n")
for i in range(len(sizeRange)):	
	filePlot.write(str(sdProfileMilliSec[i]) + " ")
filePlot.close()

"""
AutoLabel : Adds labels of the average values on the bins.
"""
def AutoLabel(rectBins, axis):
    # Attach the text labels to the bins
    for rectBin in rectBins:
        # Gets the bin height
        binHeight = rectBin.get_height()
        axis.text(rectBin.get_x() + rectBin.get_width() / 2., 1.05 * binHeight, 
        '%d'%int(binHeight), ha='center', va='bottom')

N = len(sizeRange)

# Arrange the X-axis locations for the groups
binXPos = np.arange(N)  
binWidth = 0.2       # the width of the bars

# A handle to the figure and to the axis 
figure, axis = plt.subplots()

axis.set_ylabel('Y-label')
axis.set_xlabel('X-label')
axis.set_title('Average values with the error bars')
axis.set_xticks(binXPos + binWidth)

# Latex fonts 
plt.rcParams['text.usetex']=True
plt.rcParams['text.latex.unicode']=True
plt.rcParams['font.family']='sans-serif'
plt.rcParams['font.size']=8
plt.rcParams['legend.fancybox']=True

iSize_itr = 0
for iSize in sizeRange:
	rect = axis.bar(binXPos + iSize_itr * binWidth, avgProfileMilliSec, binWidth, color = 'g', yerr = sdProfileMilliSec)
	AutoLabel(rect, axis)

plt.show()



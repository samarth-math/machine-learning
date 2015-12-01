
import os
import math

def readFileRecs(fileName):
	labelVectorPairs=[];
	f=open(fileName, 'r')
	for line in f:
		tokens = line.split(' ',1)
		vectorArr= tokens[1].split()
		vectorDict={}
		for val in vectorArr:
			keyVal = val.split(':',1)
			vectorDict.update({int(keyVal[0]):float(keyVal[1])})
		labelVectorPairs.append((int(tokens[0]),vectorDict))
	f.close()
	return labelVectorPairs

def writeFile(fileName, labelVectorPairs):
	f=open(fileName,'w')
	line=''
	for lvp in labelVectorPairs:
		if(line!=''):
			line+='\n'+str(lvp[0])
		else:
			line = str(lvp[0])
		for key in lvp[1]:
			line+=' '+str(key)+':'+str(lvp[1][key])

	f.write(line)
	
def distanceFromOrigin(xVec):
	distance=0
	for i in xVec:
		distance+=xVec[i]*xVec[i];
	return math.sqrt(distance)

def maxDistance(recs):
	maxDistance = 0
	for row in recs:
		xVec = row[1]
		dfo = distanceFromOrigin(xVec)
		if(dfo>maxDistance):
			maxDistance=dfo
	return maxDistance


def transformFeature(xVec):
	transform = {}
	count = 1
	for i in range(1,len(xVec)+1):
		for j in range(i,len(xVec)+1):
			transform[count] = xVec[i] * xVec[j]
			count+=1
	return transform

def transformFeatures(recs):
	newRow=[];
	for row in recs:
		xVec = row[1]
		newRow.append((row[0],transformFeature(xVec)))
	return newRow

def mainFunction():
	recsO= readFileRecs('astro/original/train')
	recsO1= readFileRecs('astro/original/test')
	recsS= readFileRecs('astro/scaled/train')
	recsS1= readFileRecs('astro/scaled/test')
	recsOT=transformFeatures(recsO)
	recsOT1=transformFeatures(recsO1)
	recsST=transformFeatures(recsS)
	recsST1=transformFeatures(recsS1)

	if not os.path.exists(os.path.dirname('astro/original.transformed/train')):
		os.makedirs(os.path.dirname('astro/original.transformed/train'))

	if not os.path.exists(os.path.dirname('astro/scaled.transformed/train')):
		os.makedirs(os.path.dirname('astro/scaled.transformed/train'))

	writeFile('astro/original.transformed/train',recsOT)
	writeFile('astro/original.transformed/test',recsOT1)
	writeFile('astro/scaled.transformed/train',recsST)
	writeFile('astro/scaled.transformed/test',recsST1)

	print "Max distance astro original ", maxDistance(recsO)
	print "Max distance astro scaled ", maxDistance(recsS)
	print "Max distance astro original transformed ", maxDistance(recsOT)
	print "Max distance astro scaled transformed", maxDistance(recsST)
	#for row in recs1:
	#	xVec = row[1]
	#	row[1] = transformFeature(xVec)

	#print recs1[0]


mainFunction()
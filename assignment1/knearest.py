#import numpy as np
from collections import Counter

def calcDistance (point1, point2): # record1, record2
	distance = 0
	for i in range(len(point1)-1):
		if point1[i]!=point2[i]:
			distance+=1
	return distance


def readFileRecs(fileName, start=None ,many=None, end=None):
	ticTac = []
	if many!=None: # start and End shouldn't be none either
		if start==None:
			print "start is none, but many is true. This just wrong man."
			return None
		elif end==None:
			curr=start
			for i in xrange(5):
				f=open(fileName+str(curr+1)+".txt", 'r')
				for line in f:
					lineArray = line.strip('\n').split(",")
					ticTac.append(lineArray)
				curr=(curr+1)%6
		else:
			for i in range(start,end):
				f=open(fileName+str(i)+".txt", 'r')
				for line in f:
					lineArray = line.strip('\n').split(",")
					ticTac.append(lineArray)
	else:
		f=open(fileName, 'r')
		for line in f:
			lineArray = line.strip('\n').split(",")
			ticTac.append(lineArray)
	
	#npTicTac = np.array(ticTac)
	return ticTac

# vary range of k

def insertInArray(element,position, arr):
	for i in range(len(arr)-2,position-1,-1):
		arr[i+1]=arr[i]
	arr[position]=element

def findKPoints(ksize,trainData, testPoint): # assumption : ksize will be less than datasize
	klist=[]
	 # minDist = integer, pointDes = point
	for i in range(ksize): # initialize Kmin
		minPoint = {}
		minPoint['minDist'] = calcDistance(testPoint, trainData[i])
		minPoint['pointDes'] = trainData[i]
		klist.append(minPoint)

	kmin = sorted(klist, key=lambda k: k['minDist']) #sorted list of points. Now replace as you find  smaller points
	
	for i in range(ksize,len(trainData)):
		distance = calcDistance(testPoint,trainData[i])
		kminIter=ksize
		while kminIter>0 and distance<kmin[kminIter-1]['minDist']:
			kminIter-=1
		
		if kminIter<ksize:
			minPoint={}
			minPoint['minDist'] = distance
			minPoint['pointDes'] = trainData[i]
			insertInArray(minPoint,kminIter,kmin)

	return kmin

def findKResult(kminArr):
	kmax = Counter(minPoint['pointDes'][9] for minPoint in kminArr).most_common(1)
	return kmax[0][0]

def learnK():
	dictOfKAccs = {1:[], 2:[], 3:[], 4:[], 5:[]}
	for i in range(6):
		tFileNum=(i+5)%6
		trainData = readFileRecs("tic-tac-toe/tic-tac-toe-train-",i,'true')
		testData = readFileRecs("tic-tac-toe/tic-tac-toe-train-"+str(tFileNum+1)+".txt")

		
		for ksize in range(1,6):
			count = 0
			for j in range(len(testData)):
				kmin = findKPoints(ksize,trainData,testData[j])
				label = findKResult(kmin)
				if label==testData[j][9]:
					count+=1
			accuracy = float(count)/len(testData)
			dictOfKAccs[ksize].append(accuracy)

	for k in dictOfKAccs:
		dictOfKAccs[k]=sum(dictOfKAccs[k])/6

	print "average accuracy for each value of k ",dictOfKAccs
	return max(dictOfKAccs, key=dictOfKAccs.get)
	# outer most loop, i to i+6 for the files
	# next loop k = 1 to 5
	# k min, find min distance points
	# find prediction
	# tally prediction 

def implementKNN(Kval):
	trainData = readFileRecs("tic-tac-toe/tic-tac-toe-train-",1,'true',7)
	testData = readFileRecs("tic-tac-toe/tic-tac-toe-test.txt")

	count=0
	for i in range(len(testData)):
		kmin = findKPoints(Kval,trainData,testData[i])
		label = findKResult(kmin)
		if label==testData[i][9]:
			count+=1

	accuracy = float(count)/len(testData)
	return accuracy

def mainFunction():
	k = learnK()
	print "accuracy on test data ", implementKNN(k)

mainFunction()
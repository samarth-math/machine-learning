def readFileRecs(fileName):
	dt = np.dtype([('p0'),('p1'),('p2'),('p3'),('p4'),('p5'),('p6'),('p7'),('p8'),('win')])
	readArray = np.fromfile(fileName,d)
	return readArray

	
def readFileRecs2(fileName):
	ticTac = {}

	rowN = 0
	f=open(fileName, 'r')
	
	for line in f:
		lineArray = line.strip('\n').split(",")
		row={}
		for i in range(len(lineArray)):
			row.update({i:lineArray[i]})

		ticTac.update({rowN:row})
		rowN+=1
	return ticTac


def id3Funct(data, labelPos, validAttributes=None): #data must consist of only valid rows
	#totalEntropy =  getEntropy(data, labelPos) - you don't actually need information gain, you just need minimum entropy
	if(validAttributes!=None):
		minE = getEntropy(data, validAttributes[0])
		minVal = validAttributes[0]
		for val in validAttributes:
			currEntropy = getEntropy(data, val)
			if currEntropy<minE:
				minE=currEntropy
				minVal = val
		return minVal
	else: # validAttributes = None implies, that it's the first iteration
		numrows, numcols = data.shape
		minE = getEntropy(data, 0)
		minVal = 0
		for val in range(numcols-1):
			currEntropy = getEntropy(data, val)
			if currEntropy<minE:
				minE=currEntropy
				minVal = val
		return minVal


def readFileRecs(fileName):
	labelVectorPairs=[];
	f=open(fileName, 'r')
	for line in f:
		tokens = line.split(' ',1)
		vectorArr= tokens[1].split()
		vectorDict={}
		for val in vectorArr:
			keyVal = val.split(':',1)
			vectorDict.update({keyVal[0]:keyVal[1]})
		labelVectorPairs.append(vectorDict,token[0])
	return labelVectorPairs

def readFileRecs(fileName):
	labelVectorPairs=[];
	f=open(fileName, 'r')
	for line in f:
		tokens = line.split(' ',1)
		vectorArr= tokens[1].split()
		vecArr = [float(val.split(':',1)[1]) for val in vectorArr]
		labelVectorPairs.append((vecArr,int(tokens[0])))
	return labelVectorPairs


def test():
	rateArr=[1,3,5,6,7]
	rateDict = {'yo':1, 'yolo':3, 'yoso':5, 'yoko':6, 'yono':7}

	for val in rateArr:
		print val

	for kal in rateDict:
		print kal

test()
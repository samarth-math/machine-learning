import numpy as np
from math import log
import fileinput

#assumed last column to be label
# read file as a numpy array
def readFileRecs(fileName, many=None):
	ticTac = []
	if many!=None:
		for i in range(1,7):
			f=open(fileName+str(i)+".txt", 'r')
			for line in f:
				lineArray = line.strip('\n').split(",")
				ticTac.append(lineArray)
	else:
		f=open(fileName, 'r')
		for line in f:
			lineArray = line.strip('\n').split(",")
			ticTac.append(lineArray)
	
	npTicTac = np.array(ticTac)
	return npTicTac

# DECISION TREE Data Type : python Dictionary


#Node Names : positions : 0,1,2,3,4,5,6,7,8 win:9
# currentNode is a number
# data is a 2D array of all the data

#easy to change Node choosing function
def chooseNode(data,labelCol,validAttributes):
		return id3Funct(data,labelCol,validAttributes)


# get counts for each type of label
def getRatios(column):
	uniqueVals = np.unique(column)
	ratios = {}
	total = 0
	for val in uniqueVals:
		count = list(column).count(val)
		total += count
		ratios.update({val:count})
	maxVal = max(ratios)
	ratios.update({'total':total})
	return ratios, maxVal

def getColumn(data, colPos):
	return data[:,colPos]


def entropyFunction(dataSubset,labelColN=9):
	labelCol = getColumn(dataSubset, labelColN)
	ratios = getRatios(labelCol)[0]
	sumofVal=0.0
	for val in ratios:
		if val!="total":
			p = float(ratios[val])/ratios["total"]
			term = -1*p*log(p,2)
			sumofVal+=term
	return sumofVal

def getEntropy (data, colPos):
	columnVals = getColumn(data, colPos)
	ratios = getRatios(columnVals)[0]
	sumE = 0.0
	for val in ratios:
		if val!="total":
			dataSubset = data[[data[:,colPos]==val]]
			prob = float(ratios[val])/ratios["total"]
			entropySmall =  entropyFunction(dataSubset)
			sumE+= prob*entropySmall
	return sumE

def id3Funct(data, labelPos, validAttributes=None): #data must consist of only valid rows
	#totalEntropy =  getEntropy(data, labelPos) - you don't actually need information gain, you just need minimum entropy
	minE = getEntropy(data, validAttributes[0])
	minVal = validAttributes[0]
	for val in validAttributes:
		currEntropy = getEntropy(data, val)
		if currEntropy<minE:
			minE=currEntropy
			minVal = val
	return minVal

	
def makeDecisionTree (data,labelColN,mostFrequent,validAttributes=None):
	
	# Make base conditions
	currentData = data

	if validAttributes==None:
		numrows, numcols = currentData.shape
		validAttributes = [i for i in range((numcols-1))]

	if len(currentData)<1 or len(validAttributes)<1: # if out of data, or attribute values
		return mostFrequent

	
	labelCol = getColumn(currentData,9)
	ratios = getRatios(labelCol)[0]
	subTree = {}

	if len(ratios)==2: # implies, only a total and one label value in the dictionary
		if ratios.keys()[0]=="total":
			return ratios.keys()[1]
		else:
			return ratios.keys()[0]


	currentNode=chooseNode(currentData, labelColN, validAttributes) # ID3 function
	validAttributes.remove(currentNode)
	currentNodeCol = getColumn(currentData,currentNode)
	currentNodeValues = getRatios(currentNodeCol)[0]

	
	for val in currentNodeValues:
		if val!="total":
			dataSubset = currentData[[currentData[:,currentNode]!=val]]
			subTree[val]= makeDecisionTree(dataSubset,9,mostFrequent,validAttributes)
	
	theTree={}		#theTree[str(currentNode)+"."+val] = makeDecisionTree(dataSubset,9,mostFrequent,validAttributes)
	theTree[currentNode]=subTree;
	return theTree

def testTree(record, tree, mostFrequent, unparsedAtt=None):
	if unparsedAtt==None:
		unparsedAtt = [i for i in range(len(record))]

	#print unparsedAtt
	#if len(unparsedAtt)<1:
	#	return mostFrequent

	currentNode = int(tree.keys()[0])
	currNodeVal=record[currentNode]
	unparsedAtt.remove(currentNode)
	
	if type(tree[currentNode][currNodeVal])!=dict:
		return tree[currentNode][currNodeVal]
	else:
		return testTree(record,tree[currentNode][currNodeVal],unparsedAtt)
#def useDecisionTree():




def mainFunction():
	ticTac = readFileRecs("tic-tac-toe/tic-tac-toe-train-",'true')
	testTicTac = readFileRecs("tic-tac-toe/tic-tac-toe-test.txt")
	#print
	labelCol = getColumn(ticTac,9)
	mostFrequent = getRatios(labelCol)[1]
	tree = makeDecisionTree(ticTac,9,mostFrequent)
	count=0
	for record in testTicTac:
		result = testTree(record,tree,mostFrequent)
		if result==record[9]:
			count+=1

	print "accuracy = ", float(count)/len(testTicTac)

mainFunction();


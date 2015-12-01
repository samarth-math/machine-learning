import numpy as np
import random
import matplotlib.pyplot as plt
numRuns = 5# number of crossvalidation data sampling runs


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
		vectorDict.update({0:1})
		labelVectorPairs.append((int(tokens[0]),vectorDict))

	return labelVectorPairs


def dotFunction(d1,d2) : # [{j:mij..}..][{i:mij..}..]
	val=0
	if len(d1)==len(d2):
		for i in d1:
			if(i in d2):# should always be
				val+=d2[i]*d1[i]
			else:
				print "unequal: this condition should never have happened."
		return val
	else:
		print "Found different dimensions of the two vectors being multiplied"
	exit(1)

def multiply(v,d):
	x={}
	for i in d:
		x[i]=d[i]*v;
	return x;

def sumByKey(d1,d2):
	newD={};
	for i in d1:
		if(i in d2):# should always be
			newD.update({i:d1[i]+d2[i]})
		else:
			print "the code should never be here"
	return newD

def shuffleRecords(records):
	random.shuffle(records)

def dotProd(xVec,wVec=None):
	# make weight vector
	if wVec==None:
		wVec={};
		wVec = {i:random.random() for i in range(1,len(xVec))}
		wVec[0]=1.0;
	product = dotFunction(wVec,xVec);
	return product,wVec

def updateWvec(xVec,wVec, rate, label):
	newX = multiply(rate,xVec)
	newerX=multiply(label,newX)
	updateWvec = sumByKey(newerX,wVec)
	return updateWvec

def trainPerceptron(rate, records,wVec = None,epochs=10):
	totalMistakes=0
	for i in range(epochs):
		if i!=0:
			shuffleRecords(records)

		for val in records:
			xVec = val[1]
			label = val[0]
			prod, wVec = dotProd(xVec,wVec)
			if label*(prod)<=0:
				totalMistakes+=1
				wVec = updateWvec(xVec,wVec,rate,label)

	return wVec, totalMistakes;

def trainMarginPerceptron(rate, records,mu,wVec = None,epochs=10):
	totalMistakes=0
	for i in range(epochs):
		if i!=0:
			shuffleRecords(records)
		for val in records:
			xVec = val[1]
			label = val[0]
			prod, wVec = dotProd(xVec,wVec)
			if label*(prod)<=mu:
				totalMistakes+=1
				wVec = updateWvec(xVec,wVec,rate,label)
	return wVec, totalMistakes;

def trainAggressiveMarginPerceptron(records,mu,wVec = None,epochs=10):
	totalMistakes=0
	for i in range(epochs):
		if i!=0:
			shuffleRecords(records)
		for val in records:
			xVec = val[1]
			label = val[0]
			prod, wVec = dotProd(xVec,wVec)
			ywtx=label*prod
			if ywtx<=mu:
				totalMistakes+=1
				xtx=dotProd(xVec,xVec)[0]
				rate = mu-ywtx/(xtx+1);
				wVec = updateWvec(xVec,wVec,rate,label)
	return wVec, totalMistakes;

def testAnyPerceptron(records,wVec):
	correct=0
	for val in records:
		xVec=val[1]
		label=val[0]
		prod = dotProd(wVec,xVec)[0]
		if label*prod>0:#correct
			correct+=1
	accuracy = float(correct)/len(records)
	return accuracy

def selectHyperParams(records,funcName,hyperParams=None,hyperParams2=None,wVec=None,epochs=10):
	if(funcName=='trainPerceptron'): # Assume hyper params to be a rates array
		print "Finding best hyper parameters for perceptron from given list..."
		testSetLen = len(records)/numRuns
		rateAccuracyDict = {i:0.0 for i in hyperParams}
		if hyperParams!=None:
			for val in hyperParams:
				for i in range(numRuns):
					testStart = (i)*testSetLen
					testEnd = ((i+1))*testSetLen
					testSet=records[testStart:testEnd]
					trainSet = records[testEnd:]+records[:testStart]
					wVec = trainPerceptron(val,trainSet,wVec,epochs)[0]
					accuracy = testAnyPerceptron(testSet,wVec)
					rateAccuracyDict[val]+=accuracy
				rateAccuracyDict[val]=rateAccuracyDict[val]/numRuns
		bestHP = max(rateAccuracyDict, key=rateAccuracyDict.get)
		print "Accuracy for best Hyper param", bestHP, ":",rateAccuracyDict[bestHP]
		return bestHP
	elif(funcName=='trainMarginPerceptron'):# Assume hyper params to be a (rate, mu) pair array
		print "Finding best hyper parameters for margin perceptron from given list..."
		testSetLen = len(records)/numRuns
		rateAccuracyDict = {}
		if hyperParams!=None and hyperParams2!=None:
			for rateVal in hyperParams:
				for muVal in hyperParams2:
					for i in range(numRuns):
						testStart = (i)*testSetLen
						testEnd = ((i+1))*testSetLen
						testSet=records[testStart:testEnd]
						trainSet = records[testEnd:]+records[:testStart]
						wVec = trainMarginPerceptron(rateVal,trainSet,muVal,wVec,epochs)[0]
						accuracy = testAnyPerceptron(testSet,wVec)
						paramSet = (rateVal,muVal)
						if paramSet not in rateAccuracyDict:
							rateAccuracyDict.update({paramSet:accuracy})
						else:
							rateAccuracyDict[paramSet]+=accuracy
					rateAccuracyDict[paramSet]=rateAccuracyDict[paramSet]/numRuns
		bestHP = max(rateAccuracyDict, key=rateAccuracyDict.get)
		print "Accuracy for best Hyper param", bestHP, ":",rateAccuracyDict[bestHP]
		return bestHP
	elif(funcName=='trainAggressiveMarginPerceptron'):# Assume hyper params to be a mu array
		print "Finding best hyper parameters for margin perceptron from given list..."
		testSetLen = len(records)/numRuns
		rateAccuracyDict = {i:0.0 for i in hyperParams}
		if hyperParams!=None:
			print "sampling dataSpace",
			for val in hyperParams:
				for i in range(numRuns):
					testStart = (i)*testSetLen
					testEnd = ((i+1))*testSetLen
					testSet=records[testStart:testEnd]
					trainSet = records[testEnd:]+records[:testStart]
					wVec = trainAggressiveMarginPerceptron(trainSet,val,wVec,epochs)[0]
					accuracy = testAnyPerceptron(testSet,wVec)
					rateAccuracyDict[val]+=accuracy
				rateAccuracyDict[val]=rateAccuracyDict[val]/numRuns
		bestHP = max(rateAccuracyDict, key=rateAccuracyDict.get)
		print "Accuracy for best Hyper param", bestHP, ":",rateAccuracyDict[bestHP]
		return bestHP
	else:
		print "didn't match any"


def mainFunction():
	print "===================Experiment Part 1=================table2===================\n"
	print "Running Sanity check"
	recs= readFileRecs('data0/table2')
	print "Rate : 3"
	finalWeight , mistake = trainPerceptron(3,recs,None,1)
	print "Final weights: ",finalWeight
	print "Total Mistakes: ",mistake,"\n"

	print '===================Experiment Part 2==========data0/train0.10 & test0.10================='
	recs= readFileRecs('data0/train0.10')
	recs1= readFileRecs('data0/test0.10')

	trialRates=[4,3,2.5,1.5,2.3]
	#bestHP = selectHyperParams(recs,'trainPerceptron',trialRates,None, None,1)
	#print "Selected best hyper param: ",bestHP
	wvec=None;
	mistakes=None;
	wVec,mistakes = trainPerceptron(3,recs,None,1)
	accuracy1 = testAnyPerceptron(recs1,wVec)
	accuracy = testAnyPerceptron(recs,wVec)
	print "___PERCEPTRON___ one pass result with crossvalidation for choosing hyper param"
	print "number of mistakes to train", mistakes
	print "accuracy on test Set", accuracy1
	print "accuracy on training Set", accuracy,"\n"

	trialRates=[4,3,2.5,1.5,2.3]
	trialMargins=[0.01,0.1,0.5,0.9,1.5,2,2.3]
	#bestHP = selectHyperParams(recs,trainMarginPerceptron,trialRates,trialMargins, None,1)
	#print "Selected best hyper param: ",bestHP
	wVec,mistakes = trainMarginPerceptron(3,recs,0.4,None,1)
	accuracy1 = testAnyPerceptron(recs1,wVec)
	accuracy = testAnyPerceptron(recs,wVec)
	print "___MARGIN PERCEPTRON___ one pass result with crossvalidation for choosing hyper param"
	print "number of mistakes to train", mistakes
	print "accuracy on test Set", accuracy1
	print "accuracy on training Set", accuracy,"\n"

	print"===================Experiment Part 3======Batch Perceptron==============\n"
	wVec=None;
	trialRates=None;
	bestHP=None;
	for fnum in range(2):
		print "______________FOLDER DATA",fnum,"_____________"
		listOfPercAccs=[]
		listOfPercMistakes=[]
		listOfMPercAccs=[]
		listOfMPercMistakes=[]
		listOfVectorLen = []
		listOfAMPercAccs=[]
		listOfAMPercMistakes=[]
		for dSet in range(1,11):
			recs = readFileRecs('data'+str(fnum)+'/train'+str(fnum)+'.'+str(dSet)+'0')
			recs1 = readFileRecs('data'+str(fnum)+'/test'+str(fnum)+'.'+str(dSet)+'0')
			listOfVectorLen.append(len(recs[0][1]))

			trialRates=[4,3,2.5,1.5,2.3]
			print "------------------------------------------------------------------"
			print 'data'+str(fnum)+'/train'+str(fnum)+'.'+str(dSet)+'0'
			print 'data'+str(fnum)+'/test'+str(fnum)+'.'+str(dSet)+'0'
			print "------------------------------------------------------------------"

			print "___PERCEPTRON___"
			bestHP = selectHyperParams(recs,'trainPerceptron',trialRates,None,None,1)
			wVec,mistakes = trainPerceptron(2,recs)
			print "perceptron trained in ",mistakes,"mistakes"
			accuracy1 = testAnyPerceptron(recs1,wVec)
			accuracy = testAnyPerceptron(recs,wVec)
			print "accuracy on test Set", accuracy1
			print "accuracy on training Set", accuracy,"\n"
			listOfPercAccs.append(accuracy1)
			listOfPercMistakes.append(mistakes)

			trialRates=[4,3,2.5,1.5]
			trialMargins=[0.01,0.1,0.5,0.9,1.5,2,2.3]
			print "___MARGIN PERCEPTRON___"
			bestHP = selectHyperParams(recs,'trainMarginPerceptron',trialRates,trialMargins,None,1)
			wVec,mistakes = trainMarginPerceptron(2,recs,0.01)
			print "MARGIN perceptron trained in ",mistakes,"mistakes"
			accuracy1 = testAnyPerceptron(recs1,wVec)
			accuracy = testAnyPerceptron(recs,wVec)
			print "accuracy on test Set", accuracy1
			print "accuracy on training Set", accuracy,"\n"
			listOfMPercAccs.append(accuracy1)
			listOfMPercMistakes.append(mistakes)

			mus=[0.01,0.1,0.5,0.9,1.5,2,2.3]
			print "___AGGRESSIVE MARGIN PERCEPTRON___"
			bestHP = selectHyperParams(recs,'trainAggressiveMarginPerceptron',mus,None,None,1)
			wVec,mistakes = trainAggressiveMarginPerceptron(recs,0.01)
			print "AGGRESSIVE MARGIN perceptron trained in ",mistakes,"mistakes"
			accuracy1 = testAnyPerceptron(recs1,wVec)
			accuracy = testAnyPerceptron(recs,wVec)
			print "accuracy on test Set", accuracy1
			print "accuracy on training Set", accuracy,"\n"
			listOfAMPercAccs.append(accuracy1)
			listOfAMPercMistakes.append(mistakes)

		plt.subplot(4,3,1+(6*fnum)).set_title('paccuracy')
		plt.plot(listOfVectorLen,listOfPercAccs,'r')
		plt.subplot(4,3,4+(6*fnum)).set_title('pmistakes')
		plt.plot(listOfVectorLen,listOfPercMistakes,'b')
		plt.subplot(4,3,2+(6*fnum)).set_title('maccuracy')
		plt.plot(listOfVectorLen,listOfMPercAccs,'r')
		plt.subplot(4,3,5+(6*fnum)).set_title('mmistakes')
		plt.plot(listOfVectorLen,listOfMPercMistakes,'b')
		plt.subplot(4,3,3+(6*fnum)).set_title('amaccuracy')
		plt.plot(listOfVectorLen,listOfAMPercAccs,'r')
		plt.subplot(4,3,6+(6*fnum)).set_title('ammistakes')
		plt.plot(listOfVectorLen,listOfAMPercMistakes,'b')
	plt.show()


mainFunction()

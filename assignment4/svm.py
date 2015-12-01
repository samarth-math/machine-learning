import numpy.random as random
import operator

numRuns = 10# number of crossvalidation data sampling runs


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
		if int(tokens[0])==0:
			tokens[0]=-1
		labelVectorPairs.append((int(tokens[0]),vectorDict))
	f.close()
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

def diffByKey(d1,d2):
	newD={};
	for i in d1:
		if (i in d2):# should always be
			diff = d1[i]-d2[i]
			newD.update({i:diff})
		else:
			print "the code should never be here"
	return newD

def shuffleRecords(records):
	random.shuffle(records)

def dotProd(xVec,wVec=None):
	# make weight vector
	if wVec==None:
		wVec={};
		wVec = {i:0 for i in xVec}
	product = dotFunction(wVec,xVec)
	return product,wVec


def trainSVM(records,C,irate,wVec = None,epochs=10):
	t=0
	for i in range(epochs):
		if i!=0:
			shuffleRecords(records)
		for rec in records:
			xVec = rec[1]
			label = rec[0]
			rate = float(irate)/(1+((irate*t)/C))
			WtX, wVec = dotProd(xVec,wVec)
			yWtX = label * WtX
			Cy = label * C;
			CyX = multiply(Cy,xVec)
			if yWtX<=1:
				gradient = diffByKey(wVec,CyX)
			else:
				gradient = wVec
			rdeltaE = multiply(rate,gradient)
			wVec= diffByKey(wVec, rdeltaE)
			t+=1
		print int(dotProd(wVec,wVec)[0])
	return wVec

def testSVM(records,wVec):
	correct=0
	minPosMargin=10000000
	minNegMargin =10000000
	count=0
	wsq = dotProd(wVec,wVec)[0]
	for val in records:
		count+=1
		xVec=val[1]
		label=val[0]
		#if type(xVec)==float:
		prod = dotProd(xVec,wVec)[0]
		yWtX = label*prod
		margin = float(yWtX)/wsq
		if yWtX>=0:#correct
			correct+=1
			if margin<minPosMargin:
				minPosMargin = margin
		else:
			if margin<minNegMargin:
				minNegMargin = margin

	accuracy = float(correct)/len(records)
	if minPosMargin==10000000:
		minPosMargin=0
	if minNegMargin==10000000:
		minNegMargin=0
	return accuracy, minPosMargin, minNegMargin

def selectHyperParams(records,rate,C,wVec=None,epochs=10):
	print "Finding best hyper parameters for SVM from given list..."
	if len(records)%numRuns==0:
		testSetLen = len(records)/numRuns
	else:
		testSetLen = len(records)/numRuns + 1
	rateAccuracyDict = {}
	print "sampling dataSpace",
	if rate!=None and C!=None:
		for rateVal in rate:
			for CVal in C:
				#print "new Run. rate, c",rateVal,CVal
				paramSet = (rateVal,CVal)
				for i in range(numRuns):
					testStart = (i)*testSetLen
					testEnd = ((i+1))*testSetLen
					testSet=records[testStart:testEnd]
					trainSet = records[testEnd:]+records[:testStart]
					#print testStart,testEnd
					wVec = trainSVM(trainSet,CVal,rateVal)
					accuracy = testSVM(testSet,wVec)[0]
					
					if paramSet not in rateAccuracyDict:
						rateAccuracyDict.update({paramSet:accuracy})
					else:
						rateAccuracyDict[paramSet]+=accuracy
				#print rateAccuracyDict[paramSet]/numRuns
				rateAccuracyDict[paramSet]=rateAccuracyDict[paramSet]/numRuns
		bestHP = max(rateAccuracyDict, key=rateAccuracyDict.get)
		top5 = sorted(rateAccuracyDict.iteritems(), key=operator.itemgetter(1), reverse=True)[:5]
		print "Accuracy for best Hyper param", bestHP, ":",rateAccuracyDict[bestHP]
		print "top 5 hyper params"
		for k in top5:
			print k
			#print top5[k] 
		return bestHP



def mainFunction():
	
	d0train= readFileRecs('data0/train0.10')
	d0test= readFileRecs('data0/test0.10')
	astroOrigTrain = readFileRecs('astro/original/train')
	astroOrigTest = readFileRecs('astro/original/test')
	astroScaledTrain = readFileRecs('astro/scaled/train')
	astroScaledTest = readFileRecs('astro/scaled/test')
	astroOrigTransTrain =readFileRecs('astro/original.transformed/train')
	astroOrigTransTest =readFileRecs('astro/original.transformed/test')
	astroScaledTransTrain =readFileRecs('astro/scaled.transformed/train')
	astroScaledTransTest =readFileRecs('astro/scaled.transformed/test')

	trialCs=[ 1, 10,20,30]
	trialRates = [0.001, 0.01, 0.1, 1]

	print "-----Data0 data set-----------"
	bestRC = selectHyperParams(d0train,trialRates,trialCs)
	wVec=trainSVM(d0train,bestRC[1],bestRC[0],None,30)
	#wVec=trainSVM(d0train,10,1,None,30)
	#print wVec
	accuracy, pmargin, nmargin = testSVM(d0test,wVec)
	print "accuracy on test set", accuracy
	print "lowest positive margin", pmargin
	print "most misclassified point ",	nmargin
	print "-----Data0 data set-----------"

	print "-----astro original data set -----------"
	bestRC = selectHyperParams(astroOrigTrain,trialRates,trialCs)
	wVec=trainSVM(astroOrigTrain,bestRC[1],bestRC[0],None,30)
	#wVec=trainSVM(astroOrigTrain,10,1,None,30)
	print wVec
	accuracy, pmargin, nmargin = testSVM(astroOrigTest,wVec)
	print "accuracy on test set", accuracy
	accuracy = testSVM(astroOrigTrain,wVec)[0]
	print "accuracy on train set", accuracy
	print "lowest positive margin", pmargin
	print "most misclassified point ",	nmargin
	print "-----astro original data set -----------\n"

	print "-----astro scaled data set -----------"
	bestRC = selectHyperParams(astroScaledTrain,trialRates,trialCs)
	wVec=trainSVM(astroScaledTrain,bestRC[1],bestRC[0],None,30)
	#wVec=trainSVM(astroScaledTrain,10,1,None,30)
	print wVec
	accuracy, pmargin, nmargin = testSVM(astroScaledTest,wVec)
	print "accuracy on test set", accuracy
	accuracy = testSVM(astroScaledTrain,wVec)[0]
	print "accuracy on train set", accuracy
	print "lowest positive margin", pmargin
	print "most misclassified point ",	nmargin
	print "-----astro scaled data set -----------\n"

	print "-----astro original transformed data set -----------"
	bestRC = selectHyperParams(astroOrigTransTrain,trialRates,trialCs)
	wVec=trainSVM(astroOrigTransTrain,bestRC[1],bestRC[0],None,30)
	#wVec=trainSVM(astroOrigTransTrain,10,1,None,30)
	print wVec
	accuracy, pmargin, nmargin = testSVM(astroOrigTransTest,wVec)
	print "accuracy on test set", accuracy
	accuracy = testSVM(astroOrigTransTrain,wVec)[0]
	print "accuracy on train set", accuracy
	print "lowest positive margin", pmargin
	print "most misclassified point ",	nmargin
	print "-----astro original transformed data set -----------\n"

	print "-----astro scaled transformed data set -----------"
	bestRC = selectHyperParams(astroScaledTransTrain,trialRates,trialCs)
	wVec=trainSVM(astroScaledTransTrain,bestRC[1],bestRC[0],None,30)
	#wVec=trainSVM(astroScaledTransTrain,10,1,None,30)
	print wVec
	accuracy, pmargin, nmargin = testSVM(astroScaledTransTest,wVec)
	print "accuracy on test set", accuracy
	accuracy = testSVM(astroScaledTransTrain,wVec)[0]
	print "accuracy on train set", accuracy
	print "lowest positive margin", pmargin
	print "most misclassified point ",	nmargin
	print "-----astro scaled transformed data set -----------\n"
	
	#print wVec

mainFunction()

def testFunction():
	astroOrigTrain = readFileRecs('astro/original/train')

#testFunction()

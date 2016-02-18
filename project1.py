from Patient import Patient
from decimal import *
import csv
import math


# NAIVE BAYES CLASSIFIER
# @author - CARSON SCHAEFER
# @date - 17 February 2016
# CIS 335 - Project 1


# Create patients array
patients = []
patientsTest = []
patientsTest2 = []

# Training probabilities
trainLikeMaleT = 0 #Male given virus
trainLikeMaleF = 0 #Male not given virus
trainLikeBloodPosT = 0 #Blood pos given virus
trainLikeBloodPosF = 0 #Blood pos not given virus
trainLikeWeightT = 0 #Weight > 170 given virus
trainLikeWeightF = 0 #Weight > 170 not given virus
trainPrior = 0

trainWeightsT = [] # List of weights for people given virus
trainWeightsF = [] # List of weights for people not given virus


# Confusion matrix
truePos = 0
trueNeg = 0
falsePos = 0
falseNeg = 0

# read training data from csv file, create patient object, append to patients array
try:
	trainingData = csv.reader(open("proj1train.csv"))
	for row in trainingData:
		# sets virus to true if 'Y', sets weight to true if > 170
		temp = Patient(int(row[0]), row[1], row[2], True if int(row[3])>170 else False, True if row[4]=='Y' else False)
		patients.append(temp)
		trainWeightsT.append(int(row[3])) if row[4]=='Y' else trainWeightsF.append(int(row[3]))
except Exception as e:
	print(e)


# read test data from csv file, create patient object, append to patients array
try:
	testData = csv.reader(open("proj1test.txt"))
	for row in testData:
		# sets virus to true if 'Y', sets weight to true if > 170
		temp = Patient(int(row[0]), row[1], row[2], Decimal(row[3]), True if row[4]=='Y' else False)
		patientsTest.append(temp)
except Exception as e:
	print(e)


# read test data from csv file, create patient object, append to patients array
try:
	testData = csv.reader(open("proj1test.txt"))
	for row in testData:
		# sets virus to true if 'Y', sets weight to true if > 170
		temp = Patient(int(row[0]), row[1], row[2], True if Decimal(row[3])>170 else False, True if row[4]=='Y' else False)
		patientsTest2.append(temp)
except Exception as e:
	print(e)


# return the priors for the column in the dataset
# @col - specifies which column
# @val - value of col compared 
# possibilies are...
# @gender - Returns for gender (Male="male" , Female="female")
# @blood - Returns for blood type for val (O+-, A+-, B+-, AB+-)
# @positive - val = "+" or "-"
# @weight - Val = True for > 170 or False
# @virus - val = True or False
def priors(dataset, col, val=None):
	# map for which prior calculations specified by col
	switch = {
		"gender": Decimal(sum(p.gender==val for p in dataset))/Decimal(len(dataset)),
		# Prior for blood type entered after '-' delimeter
		"blood": Decimal(sum(p.blood==val for p in dataset))/Decimal(len(dataset)),
		# Prior for positive blood types
		"positive": Decimal(sum(str(val) in p.blood for p in dataset))/Decimal(len(dataset)),
		# has weight above 170
		"weight": Decimal(sum(p.weight==val for p in dataset))/Decimal(len(dataset)),
		# Has virus prior
		"virus": Decimal(sum(p.virus==val for p in dataset))/Decimal(len(dataset)),
	}
	return switch.get(col)


# return a subset of the dataset based on the column values
# @col - specifies which column
# @val - value of col compared 
# possibilies are...
# @gender - Returns for gender (Male="male" , Female="female")
# @blood - Returns for blood type for val (O+-, A+-, B+-, AB+-)
# @positive - val = "+" or "-"
# @weight - Val = True for > 170 or False
# @virus - val = True or False
def subset(dataset, col, val=None):
	temp = []
	if col == "gender":
		for p in dataset:
			if p.gender==val:
				temp.append(p)

	elif col == "blood":
		for p in dataset:
			if p.blood==val:
				temp.append(p)

	elif col == "positive":
		for p in dataset:
			if val in p.blood:
				temp.append(p)

	elif col == "weight":
		for p in dataset:
			if p.weight==val:
				temp.append(p)

	elif col == "virus":
		for p in dataset:
			if p.virus==val:
				temp.append(p)
	else:
		return

	return temp


# Calculate Mean
def mean(numbers):
	return sum(numbers)/Decimal(len(numbers))

# Calculate Standard Deviation, note uses N - 1
def stdDev(numbers):
	m = mean(numbers)
	return math.sqrt(sum(pow(n-m,2) for n in numbers)/Decimal(len(numbers)-1))


# Likelihood of a male given (True or False) Virus
# @virus - if they are given the virus or not
def likeMale(dataset, virus):
	return priors(subset(dataset,"virus",virus),"gender","male")


# Likelihood of individual with positive blood type given (True or False) Virus
# @virus - if they are given the virus or not
def likeBloodPos(dataset, virus):
	return priors(subset(dataset,"virus",virus),"positive","+")



# Likelihood of individual with Weight > 170 given (True or False) Virus
# @virus - if they are given the virus or not
def likeWeight(dataset, virus):
	return priors(subset(dataset,"virus",virus),"weight",True)



# Calculate probability of patient
# @virus - Calc if they have virus or not
def probability(dataset, patient, virus):
	calc = trainPrior if virus else (1-trainPrior)
	if patient.gender == "male":
		calc *= trainLikeMaleT if virus else trainLikeMaleF
	else:
		calc *= (1-trainLikeMaleT) if virus else (1 - trainLikeMaleF)

	if "+" in patient.blood:
		calc *= trainLikeBloodPosT if virus else trainLikeBloodPosF
	else:
		calc *= (1 - trainLikeBloodPosT) if virus else (1 - trainLikeBloodPosF)

	if patient.weight:
		calc *= trainLikeWeightT if virus else trainLikeWeightF
	else:
		calc *= (1 - trainLikeWeightT) if virus else (1 - trainLikeWeightF)

	print str(str(patient.id) + ": " + str(virus) + " : "+str(calc))
	return calc

# Calculate probability of patient
# @virus - Calc if they have virus or not
# THIS USES A NUMBERIC VALUE FOR WEIGHT
def probabilityNumeric(dataset, patient, virus):
	calc = trainPrior if virus else (1-trainPrior)
	if patient.gender == "male":
		calc *= trainLikeMaleT if virus else trainLikeMaleF
	else:
		calc *= (1-trainLikeMaleT) if virus else (1 - trainLikeMaleF)

	if "+" in patient.blood:
		calc *= trainLikeBloodPosT if virus else trainLikeBloodPosF
	else:
		calc *= (1 - trainLikeBloodPosT) if virus else (1 - trainLikeBloodPosF)

	if virus:
		calc *= Decimal((1/(math.sqrt(2*math.pi*math.pow(stdDev(trainWeightsT),2))))*(math.pow(math.e,(math.pow((mean(trainWeightsT)-patient.weight),2))/(2*math.pow(stdDev(trainWeightsT),2)))))
	else:
		calc *= Decimal((1/(math.sqrt(2*math.pi*math.pow(stdDev(trainWeightsF),2))))*(math.pow(math.e,(math.pow((mean(trainWeightsF)-patient.weight),2))/(2*math.pow(stdDev(trainWeightsF),2)))))

	print str(str(patient.id) + ": " + str(virus) + " : "+str(calc))
	return calc

# Compare the probability of the patient having the virus or not
# Return Actual in first col and Second in Predicted
# Calculates the confusion matrix values
def compareProb(dataset, patient):
	output = "Y - " if patient.virus else "N - "
	output = output + "Y" if probability(dataset,patient,True) > probability(dataset,patient,False) else output + "N"
	if output == "Y - Y":
		global truePos
		truePos +=1
	elif output == "N - N":
		global trueNeg
		trueNeg +=1
	elif output == "Y - N":
		global falseNeg
		falseNeg +=1
	elif output == "N - Y":
		global falsePos
		falsePos +=1

	return output


# Compare the probability of the patient having the virus or not
# Return Actual in first col and Second in Predicted
# Calculates the confusion matrix values
# USES NUMERIC PROB FOR WEIGHT ATTRIBUTE
def compareProbNumeric(dataset, patient):
	output = "Y - " if patient.virus else "N - "
	output = output + "Y" if probabilityNumeric(dataset,patient,True) > probabilityNumeric(dataset,patient,False) else output + "N"
	if output == "Y - Y":
		global truePos
		truePos +=1
	elif output == "N - N":
		global trueNeg
		trueNeg +=1
	elif output == "Y - N":
		global falseNeg
		falseNeg +=1
	elif output == "N - Y":
		global falsePos
		falsePos +=1

	return output

# print priors and likelihoods
def printPrior_Likelihoods():
	print "Prior for Virus: " + str(priors(patients,"virus", True)) + "\t\tNot Virus: " + str(priors(patients,"virus",False))
	print "Likelihood for female Virus: " + str(1 - likeMale(patients,True)) + "\t\tNot Virus: " + str(1-likeMale(patients,False))
	print "Likelihood for male Virus: " + str(likeMale(patients,True)) + "\t\tNot Virus: " + str(likeMale(patients,False))
	print "Likelihood for blood + Virus: " + str(likeBloodPos(patients, True)) + "\t\tNot Virus: " +str(likeBloodPos(patients,False)) 
	print "Likelihood for blood - Virus: " + str(1-likeBloodPos(patients, True)) + "\t\tNot Virus: " +str(1-likeBloodPos(patients,False)) 
	print "Likelihood for weight > 170 Virus: " + str(likeWeight(patients,True)) + "\t\tNot Virus: " +str(likeWeight(patients,False))
	print "Likelihood for weight < 170 Virus: " + str(1-likeWeight(patients,True)) + "\t\tNot Virus: " +str(1-likeWeight(patients,False))


# print confusion matrix
def printConfusionMatrix():
	print "\n--Confusion Matrix--"
	print "X:Predicted\tY:Actual"
	print "     Yes  No"
	print "Yes: " + str(truePos) + " " + str(falseNeg)
	print " No: " + str(falsePos) + "  " + str(trueNeg)

#Store likelihoods
def train(dataset):
	global trainLikeMaleT
	global trainLikeMaleF
	global trainLikeBloodPosT
	global trainLikeBloodPosF
	global trainLikeWeightT
	global trainLikeWeightF
	global trainPrior
	trainLikeMaleT = likeMale(dataset,True)
	trainLikeMaleF = likeMale(dataset,False)
	trainLikeBloodPosT = likeBloodPos(dataset,True)
	trainLikeBloodPosF = likeBloodPos(dataset,False)
	trainLikeWeightT = likeWeight(dataset,True)
	trainLikeWeightF = likeWeight(dataset,False)
	trainPrior = priors(dataset,"virus",True)


train(patients)

print "\n"
printPrior_Likelihoods()
print "TEST DATASET WITHOUT USING NUMERIC WEIGHT"
print "\nID - Actual - Predicted"
for p in patientsTest2:
	print str(p.id) + " - " + compareProb(patientsTest2, p)

printConfusionMatrix()

# reset confusion matrix to prepare for test data
truePos = 0
trueNeg = 0
falsePos = 0
falseNeg = 0


print "---------------TEST DATASET------------------"
print "\nID - Actual - Predicted"
for p in patientsTest:
	print str(p.id) + " - " + compareProbNumeric(patientsTest, p)

printConfusionMatrix()





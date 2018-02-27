from sklearn.ensemble import RandomForestClassifier
from parser import parse_data as parse
from random import choice
from math import ceil

TRAIN_PCT = 0.90
ATTRIBUTES = 4


data = parse('Iris.txt', data_format=[float, float, float, float, str])


# Count all possible types that the data can be classified as
classes = []
for d in range(len(data)):
	if data[d][ATTRIBUTES] not in classes: classes.append(data[d][ATTRIBUTES])
CLASSES = len(classes)


# Create a set of data with which to train our model
training_data = []
for _ in range(0, ceil(TRAIN_PCT * len(data))):
	# Find a random set of data that is not already in the training set
	d = choice(data)
	while d in training_data: d = choice(data)
	training_data.append(d)

# Test data will be all data that remains (attributes only)
test_data = [d for d in data if d not in training_data]
test_attr = [d[:ATTRIBUTES] for d in test_data]

#print(len(training_data))
#print(len(test_data))
#print(test_data)


x = [d[:ATTRIBUTES] for d in training_data]
y = [classes.index(d[ATTRIBUTES]) for d in training_data]
#print(y)


clf = RandomForestClassifier(verbose=1)

clf.fit(x, y)

predictions = clf.predict(test_attr)


for i in range(len(predictions)):
	test = test_attr[i]
	real = test_data[i][ATTRIBUTES]
	pred = classes[predictions[i]]

	print(str(test)+" is predicted to be "+str(pred)+". In reality, it is "+str(real))


from sklearn.ensemble import RandomForestClassifier
from random import choice
from math import ceil



def run_classifier(data, classes, pct_train=0.9):
	'''Runs the random forest classifier on a given dataset
		parameters:
			data---- list[list[]], required.
					 The data to train and test the classifier on.
			classes---- list, required
						A list of all possible classifications in the dataset
			pct_train---- float, optional (default=0.90)
						  The percent of data that the model will be trained on
						  The percent if data the model is tested on is (1 - pct_train)

		returns:
			tuple (correct, tests)
			correct---- The number of times the model classifies a piece of test data correctly
			tests---- The number of times the model was tested. Total percent correct is (correct / tests)
	'''

	tests = 0
	correct = 0

	# Create a set of data with which to train our model
	training_data = []
	for _ in range(0, ceil(pct_train * len(data))):
		# Find a random set of data that is not already in the training set
		d = choice(data)
		while d in training_data: d = choice(data)
		training_data.append(d)

	# Test data will be all data that remains (attributes only)
	test_data = data[:]
	for d in training_data: test_data.remove(d)
	test_attr = [d[:-1] for d in test_data]

	# Split data into features (x) and classification (y)
	x = [d[:-1] for d in training_data]
	y = [classes.index(d[-1]) for d in training_data]

	# Create a forest classifier object
	clf = RandomForestClassifier(verbose=1)

	# Train the classifier on the dataset
	clf.fit(x, y)

	# Find predictions for the test data
	predictions = clf.predict(test_attr)

	# Calculate the efficiency of the classifier and print results to the console
	for i in range(len(predictions)):
		test = test_attr[i]
		real = test_data[i][-1]
		pred = classes[predictions[i]]

		if (real == pred): 
			correct += 1
			string = "(CORRECT)\t"
		else:
			string = "(INCORRECT)\t"

		string += str(test)+" is predicted to be "+str(pred)+". In reality, it is "+str(real)

		print(string)

	tests += len(predictions)

	return (correct, tests)

	


from sklearn.ensemble import RandomForestClassifier
from random import choice
from math import ceil

from parser import parse_data as parse


DATA_FORMAT = [str, str, str, int, int, str, str]
CLASSIFICATION = 0  # Index where target is stored in data

def run_classifier(data, pct_train=0.90):
    '''Run the random forest classifier on the car dataset
        parameters:
            data---- list of lists, required
                     the parsed data that will be used to train and test the model
            pct_train---- float, optional (default=0.90)
                          The percentage of data that will be used to train the model
                          Percentage of test data is therefore 1-pct_train
            returns:
                tuple of (number of correctly classified items, number of tests run)
                If model performs perfectly, correct=tests
        '''

    # Get all possible types that the data can be classified as
    allClasses = []
    for attr in range(len(data[0])):
        # Find all classifications for this attribute
        classes = {}
        attributes = 0
        for d in range(len(data)):
            if data[d][attr] not in classes: 
                classes[data[d][attr]] = attributes
                attributes += 1
        allClasses.append(classes)


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
    test_attr = [d[1:] for d in test_data]

    linenum = 0
    for line in test_attr:
        test_attr[linenum] = get_numeric(line, allClasses)
        linenum += 1
    print(test_attr)

    # Split training data into attributes/classification
    x = []
    for line in training_data:
        x.append(get_numeric(line[1:], allClasses))

    print(x)

    y = [allClasses[0][d[0]] for d in training_data]


    # Create a forest classifier object
    clf = RandomForestClassifier()

    # Train the classifier on the dataset
    clf.fit(x, y)


    # Find predictions for the test data
    predictions = clf.predict(test_attr)

    # Calculate the efficiency of the classifier and print results to the console
    for i in range(len(predictions)):
        test = test_attr[i]
        
        real = test_data[i][CLASSIFICATION]
        pred = None

        for key in allClasses[0]:
            if allClasses[0][key] == predictions[i]:
                pred = key
                break

        if (real == pred): 
            correct += 1
            string = "(CORRECT)\t"
        else:
            string = "(INCORRECT)\t"

        string += str(test)+" is predicted to be "+str(pred)+". In reality, it is "+str(real)

        print(string)

    tests += len(predictions)

    return (correct, tests)



def get_numeric(line, allClasses):
    '''Get numeric value for a set of attributes from allClasses dictionary
        parameters:
            line---- list, required
                     the line that contains the set of attributes to match
            allClasses---- list of dictionaries, required
                           contains the numerical lookup value for each attribute
        returns:
            a list of numeric attributes that is the translated version of line
    '''

    dict_num = 1 # Skip first dictionary (the target dictionary)
    translated = []
    for attr in line:
        translated.append(allClasses[dict_num][attr])
        dict_num += 1
    return translated



#data = parse('Car.csv', header=True, data_format=DATA_FORMAT)
#run_classifier(data)
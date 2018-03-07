import csv
import math
import numpy as np

samplefilepath = 'Car.csv'
PCT_TRAIN = 0.90

#Probability is calculated, it returns the freq/length of data
def chance(val, division):
    length = len(division)
    freq = division.count(val)
    return freq/length


#Function to calculate the entropy, returns the entropy
def cal_entropy(data):
    goal = [row[0] for row in data]
    targets_freq = {i: goal.count(i) for i in set(goal)}
    #after finding the frequency of targets, entropy_by_target is found by invoking entropy_for_a_target function
    entropy_target = {i: entropy_target_val(freq, len(data)) for i, freq in targets_freq.items()}
    # the entropy value is returned
    return sum(entropy_target.values())

#Entropy value for a target is being returned to the calling function
def entropy_target_val(freq, data):
    val = (-freq/data)*math.log(freq/data,2)
    return val


#Function to calculate the information gain, returns the gain value
def info_Gain(data, index):
    #entropy function is called
    entropy_val = cal_entropy(data) # entropy value
    attr_val = [x[index] for x in data] # attribute_values
    #unique attribute values is found
    attr = set(attr_val)
    gain = 0
    for attribute_value in attr:
        #filtered data gets calculated
        new_data = [x for x in data if x[index] == attribute_value]
        #Probabaility function is called
        probability = chance(attribute_value, attr_val)
        entropy = cal_entropy(new_data)
        gain += (probability * entropy)
        #returns the gain value
    return entropy_val - gain


# Function to choose the best attribute, it returns the maximum gain attribute
def best_attribute(data):
    max_gain = 0
    max_gain_attribute = -1
    total_attributes = len(data[0]) - 1
    for x in range(total_attributes):
        attribute_index = x + 1
        #informationgain function is called to get the gain values of attributes
        gain = info_Gain(data, attribute_index)
        if gain > max_gain:
            max_gain = gain
            max_gain_attribute = attribute_index
    return max_gain_attribute

#Function to build the decision tree, it returns a tree
def build_decision_tree(data, title):
    total = len(data[0]) - 1
    if total == 0:
        return
    goal = set([i[0] for i in data])
    if len(goal) == 1:
        return goal.pop()
    #The best attribute is choosen through the choose_attribute function
    choose = best_attribute(data)
    tree = {title[choose]: {}}
    attr_val = set([i[choose] for i in data])
    for value in attr_val:
        new_data = [i for i in data if i[choose] == value]
        sub_tree = build_decision_tree(new_data, title)
        tree[title[choose]][value] = sub_tree
    return tree

#Function to test the decision tree and return the percentage of accuracy
def test_tree(data, title, goal, tree):
    targets = [row[0] for row in data]
    accurate = 0
    pred = []
    #every instance in the test data gets checked
    for index, row in enumerate(data):
        #the value from get_result_from_decision_tree is stored in result
        result_val = result(tree, row, title, goal)

        if result_val == targets[index]:
        #if the value in result equals the target value in the set it increments the no_of_correct_answers
            accurate +=1
            pred.append(targets[index])
    return accurate, len(data), pred

#Function to get the result from decision tree, it returns a tree
def result(tree, row, title, goal):
    if len(row) != len(title):
        print("R", row)
        print("T", title)
        print("PROBLEM")

    while True:
        key = list(tree.keys())[0]
        index = title.index(key)
        val = row[index]
        if key not in tree:
            return None
        if val not in tree[key]:
            return None
        tree = tree[key][val]
        if tree in list(goal):
            return tree

def run_classifier(PCT_TRAIN, verbose=True):
    #Function to read the dataset from csv file
    with open(samplefilepath, 'r') as f:
      reader = csv.reader(f)
      data = list(reader)
      title = data.pop(0)
      targets = [row[0] for row in data]
      unique_targets = set(targets)
      #random sampling with replacement
      training_row_indices = np.random.choice(len(data),int(len(data) * PCT_TRAIN), replace=True)
      X_train = [data[i] for i in training_row_indices]

      testing_row_indices = np.random.choice(len(data), int(len(data) * (1-PCT_TRAIN)), replace=True)
      X_test = [data[i] for i in testing_row_indices]
      #build_tree function is called
      tree = build_decision_tree(X_train, title)
      #accuracy of the test data is calculated through the test_decision_tree function
      accurate, total, pred = test_tree(X_test, title, unique_targets, tree)


      col1 = {'vhigh': 0, 'high': 1, 'med': 2, 'low':3}
      col2 = {'vhigh': 0, 'high': 1, 'med': 2, 'low':3}
      col3 = {'2':0, '3':1, '4':2, '5more':3}
      col4 = {'2':0, '4':1, '4':2, 'more':3}
      col5 = {'small': 0, 'med': 1, 'big': 2}
      col6 = {'low': 0, 'med':1, 'high': 2}
      listn = []

      if verbose:
          for i in range(0, len(pred)):
              listn = [col1[X_test[i][1]], col2[X_test[i][2]], col3[X_test[i][3]], col4[X_test[i][4]], col5[X_test[i][5]],
                        col6[X_test[i][6]]]
              if pred[i] == X_test[i][0]:
                  print ("(CORRECT) ", listn," is predicted to be " + pred[i] + ". In reality, it is " + X_test[i][0])
              else:
                  print("(INCORRECT) ", listn," is predicted to be " + pred[i] + ". In reality, it is " + X_test[i][0])

      #print("accurate prediciton: ", accurate, ", total prediction:", total)
      return accurate, total

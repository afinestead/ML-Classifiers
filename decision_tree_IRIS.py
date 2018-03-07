from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np

# Get sklearn IRIS data
iris=datasets.load_iris()
X=iris.data
#0 - Iris-setosa, 1 - Iris-versicolor, 2 - Iris-virginica
y=iris.target


# calculate the number of each label
def num_of_label(rows):
    number = {}  # A dictionary has label:count.
    for row in rows:
        label = row[-1]
        if label not in number:
            number[label] = 0
        number[label] += 1
    return number

#check if value is a int or a float
def check_type(val):
    return isinstance(val, int) or isinstance(val, float)

class Test:
    def __init__(self, column, value):
        self.column = column
        self.value = value

    def check(self, sample):
        val = sample[self.column]
        if check_type(val):
            return val >= self.value
        else:
            return val == self.value


# split values into two categories
def category(rows, test):
    right = []
    wrong = []
    for row in rows:
        if test.check(row):
            right.append(row)
        else:
            wrong.append(row)
    return right, wrong

# Find the impurity based on Gini
def gini(rows):
    number = num_of_label(rows)
    impurity = 1
    for label in number:
        current_label = number[label] / float(len(rows))
        impurity -= current_label**2
    return impurity

def value_gain(right, wrong, unpredict):
    p = float(len(right)) / (len(right)+len(wrong))
    result = unpredict - p * gini(right) - (1 - p) * gini(wrong)
    return result

# Find the best way to split the data
def split(rows):
    best_value_gain = 0
    best_split = None
    unpredict = gini(rows)
    data = len(rows[0]) - 1
    for col in range(data):
        values = set([row[col] for row in rows])  #only use the unique values in the column
        for val in values:
            spliting = Test(col, val)
            right, wrong = category(rows, spliting) # splitting
            if len(right) == 0 or len(wrong) == 0: # if no split occurs
                continue #tries next val in values
            gain = value_gain(right, wrong, unpredict)
            if gain >= best_value_gain:
                best_value_gain, best_split = gain, spliting

    return best_value_gain, best_split

class Leaf:
    def __init__(self, rows):
        self.predictions = num_of_label(rows)

class Decision_Node:
    #This holds a reference to the question, and to the two child nodes.
    def __init__(self, question, true_branch, false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch

def build_tree(rows):
    gain, spliting = split(rows)
    if gain == 0:
        return Leaf(rows)
    true_rows, false_rows = category(rows, spliting)
    true_branch = build_tree(true_rows)
    false_branch = build_tree(false_rows)
    return Decision_Node(spliting, true_branch, false_branch)


def classify(row, node):
    if isinstance(node, Leaf):
        return node.predictions
    if node.question.check(row):
        return classify(row, node.true_branch)
    else:
        return classify(row, node.false_branch)



def run_classifier(pct_train, verbose=True):
    # Get test and train data from complete dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1 - pct_train)
    train = np.c_[X_train, y_train]
    test = np.c_[X_test, y_test]



    my_tree = build_tree(train)
    # 0 - Iris-setosa, 1 - Iris-versicolor, 2 - Iris-virginica
    type = {0.0:"Iris-setosa", 1.0: "Iris-versicolor", 2.0: "Iris-virginica"}
    total = 0
    accurate = 0
    for row in test:
        label = list(classify(row, my_tree))
        char = ""
        if label == [0.0]:
            char = "Iris-setosa"
        if label == [1.0]:
            char = "Iris-versicolor"
        if label == [2.0]:
            char = "Iris-virginica"
        if verbose: print ((row[:-1]), " is predicted to be " + char + ". In reality, it is" + type[row[-1]])
        if char == type[row[-1]]:
            accurate+=1
        total+=1
    #print("accurate prediciton:", accurate, ", total prediction:", total)
    return accurate, total




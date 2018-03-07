'''Classifiers.py
    Developed by Alex Finestead and Pinzhu Qian, University of Washington

    Runs, tests, and compares two machine learning classifiers:
    A random forest, and a decision tree

    There are currently two data sets available: a standard iris dataset,
    and a dataset for car selection

    Usage is shown below'''



USAGE = "USAGE: python3 Classifiers.py [--classifier=(str)] [--dataset=(str)] [--trials=(int)] [--train=(float)] [--compare] [--testlearn] [--verbose] [--time]"
'''
    --classifier=random-forest/decision-tree
        default is random-forest
    --dataset=iris/cars
        default is iris
    --trials=any integer greater than 0
        default is 1
    --train=any number between 0 and 1 inclusive
        default is 0.90
    --compare
        sets whether we are comparing models or just using 1
        default is off (not comparing)
    --testlearn
        sets whether we are looking at the learning ability of the model
        tests model by training on 5% data thru 95% data
        default is off
    --verbose
        controls the verbosity of the program (printing to the console)
        default is on (yes print)
    --time
        set to compare models based on execution time
        default is off
'''


import sys
from parser import parse_data as parse

# Arguments to classifier program
CLASSIFIER = 'random-forest'
DATASET = 'iris'
N_TRIALS = 1
PCT_TRAIN = 0.90
COMPARE = False
VERBOSE = None
TEST_LEARNING = False
TIME = False


CLASSIFIERS = ['random-forest', 'decision-tree']
DATASETS = ['iris', 'cars']


def main():
    global CLASSIFIER, N_TRIALS, PCT_TRAIN, VERBOSE, TEST_LEARNING, TIME
    get_args()

    if VERBOSE is None: VERBOSE = not COMPARE
    
    # if --testlearn is used
    if TEST_LEARNING: 
        import matplotlib.pyplot as plt                 # need plotting option
        pcts = [x / 100 for x in range(5, 100, 5)]      # 0.05 -> 0.95

    if TIME: import time


    random_forest_tests = 0
    random_forest_correct = 0

    decision_tree_tests = 0
    decision_tree_correct = 0


    # If comparing (using both classifiers), or using the random forest only
    # Set by --classifier=random-forest, or with --compare
    if COMPARE or CLASSIFIER == 'random-forest':
        if DATASET == 'iris':
            try:
                from random_forest_IRIS import run_classifier as run_random_forest
            except:
                print("Cannot import random forest classifier for the iris dataset. Exiting.")
                sys.exit()

            data = parse('Iris.txt', data_format=[float, float, float, float, str])
        
        elif DATASET == 'cars':
            try:
                from random_forest_CARS import run_classifier as run_random_forest
            except:
                print("Cannot import random forest classifier for the car dataset. Exiting.")
                sys.exit()

            data = parse('Car.csv', data_format=[str, str, str, int, int, str, str])
        
        if TEST_LEARNING:
            random_forest_results = []
            for i in pcts:
                print("\nTesting random forest learning ability by training on "+str(i)+"%"+" of data")
                correct, tests = run_random_forest(data, pct_train=i, verbose=VERBOSE)
                print("\t%.2f" % (100 * correct / tests), "%"+" correct")
                random_forest_results.append((correct / tests))

        # Run the random forests classifier N_TRIALS times on data from DATASET
        else:
            if TIME: start_rf = time.time()

            for i in range(N_TRIALS):
                print("\nRunning random forest trial "+str(i+1)+" of "+str(N_TRIALS)+"...")
                correct, tests = run_random_forest(data, pct_train=PCT_TRAIN, verbose=VERBOSE)
                random_forest_correct += correct
                random_forest_tests += tests

            if TIME: rf_time = time.time() - start_rf


    # If comparing both classifiers, or classifier is decision tree
    # Set by --classifier=decision-tree, or with --compare
    if COMPARE or CLASSIFIER == 'decision-tree':
        if DATASET == 'iris':
            try:
                from decision_tree_IRIS import run_classifier as run_decision_tree
            except:
                print("Cannot import decision tree classifier for the iris dataset. Exiting.")
                sys.exit()


        elif DATASET == 'cars':
            try:
                from decision_tree_CARS import run_classifier as run_decision_tree
            except:
                print("Cannot import decision tree classifier for the car dataset. Exiting.")
                sys.exit()

        
        if TEST_LEARNING:
            decision_tree_results = []
            for i in pcts:
                print("\nTesting decision tree learning ability by training on "+str(i)+"%"+" of data")
                correct, tests = run_decision_tree(i, verbose=VERBOSE)
                print("\t%.2f" % (100 * correct / tests), "%"+" correct")
                decision_tree_results.append((correct / tests))

        # Run the classifier N_TRIALS times
        else:
            if TIME: start_dt = time.time()

            for i in range(N_TRIALS):
                print("\nRunning decision tree trial "+str(i+1)+" of "+str(N_TRIALS)+"...")
                correct, tests = run_decision_tree(PCT_TRAIN, verbose=VERBOSE)
                decision_tree_correct += correct
                decision_tree_tests += tests

            if TIME: dt_time = time.time() - start_dt
        

    # If we ran the random forest classifier, print the results to the console
    if random_forest_tests != 0:
        print("\nFINAL RESULT FOR RANDOM FOREST CLASSIFIER:")
        print("\t"+str(random_forest_correct)+"/"+str(random_forest_tests)+" were correctly predicted ("\
                  +str(100*random_forest_correct/random_forest_tests)+"%)")
        if TIME: print("\tCompleted "+str(random_forest_tests)+" in %.2f" % (rf_time)+" seconds")


    # If we ran the decision tree classifier, print the results to the console
    if decision_tree_tests != 0:
        print("\nFINAL RESULT FOR DECISION TREE CLASSIFIER:")
        print("\t"+str(decision_tree_correct)+"/"+str(decision_tree_tests)+" were correctly predicted ("\
                  +str(100*decision_tree_correct/decision_tree_tests)+"%)")
        if TIME: print("\tCompleted "+str(decision_tree_tests)+" in %.2f" % (dt_time)+" seconds")


    if TEST_LEARNING:
        if COMPARE or CLASSIFIER == 'random-forest':
            plt.scatter(pcts, random_forest_results, color='r', label='Random Forest Classifier')
        if COMPARE or CLASSIFIER == 'decision-tree':
            plt.scatter(pcts, decision_tree_results, color='b', label='Decision Tree Classifier')
        plt.xlabel("%" " of data used for training")
        plt.ylabel("%"+" of trials predicted correctly")
        plt.legend(loc=4)   # Put legend in lower right corner (4)
        plt.show()


    if COMPARE and not TEST_LEARNING:
        if random_forest_correct == decision_tree_correct:
            print("\nBoth classifiers performed equally across "+str(N_TRIALS)+" independent trials ("+str(decision_tree_tests)+" tests total)\n")
        elif random_forest_correct > decision_tree_correct:
            print("\nThe random forest classifier outperformed the decision tree classifier by "+str(random_forest_correct - decision_tree_correct)+\
                  " tests out of "+str(decision_tree_tests)+" total\n")
            print("The random forest classifier performed %.2f" % (100*(random_forest_correct - decision_tree_correct) / decision_tree_tests)+\
                  "% better than the decision tree classifier\n")
        else:
            print("\nThe decision tree classifier outperformed the random forest classifier by "+str(decision_tree_correct - random_forest_correct)+\
                  " tests out of "+str(decision_tree_tests)+" total\n")
            print("The decision classifier performed %.2f" % (100*(decision_tree_correct - random_forest_correct) / decision_tree_tests)+\
                  "% better than the random forest classifier\n")

        if TIME:
            if rf_time < dt_time:
                print("The random forest classifer ran %.2f" % (dt_time / rf_time)+" times faster than the decision tree classifier")
            else:
                print("The decision tree classifer ran %.2f" % (rf_time / dt_time)+" times faster than the random forest classifier")
            print("\n")





def get_args():
    '''Parse any command line arguments to the program
        parameters:
            None
        returns:
            None (stored in globals)
    '''

    global CLASSIFIER, N_TRIALS, PCT_TRAIN, DATASET, COMPARE, VERBOSE, TEST_LEARNING, TIME

    for arg in sys.argv:
        if arg.startswith('--classifier='): 
            try: CLASSIFIER = arg[arg.index('=')+1:]
            except: print_usage_err()
            if CLASSIFIER not in CLASSIFIERS: 
                print("Error: classifier \'"+CLASSIFIER+"\' is not a valid classifier")
                print("Classifier must be one of the following:")
                for c in CLASSIFIERS: print("\t"+c)
                print_usage_err()
        elif arg.startswith('--trials='): 
            try: N_TRIALS = int(arg[arg.index('=')+1:])
            except: print_usage_err()
            if N_TRIALS <= 0: 
                print("Error: number of trials must be greater than 0")
                print_usage_err()
        elif arg.startswith('--train'): 
            try: PCT_TRAIN = float(arg[arg.index('=')+1:])
            except: print_usage_err()
            if PCT_TRAIN > 1 or PCT_TRAIN < 0:
                print("Error: train must be between 0.0 and 1.0, inclusive")
                print_usage_err()
        elif arg.startswith('--dataset='): 
            try: DATASET = arg[arg.index('=')+1:]
            except: print_usage_err()
            if DATASET not in DATASETS: 
                print("Error: dataset \'"+DATASET+"\' is not a valid dataset")
                print("Dataset must be one of the following:")
                for d in DATASETS: print("\t"+d)
        elif arg == '--compare': COMPARE = True
        elif arg == '--verbose': VERBOSE = True
        elif arg == '--testlearn': TEST_LEARNING = True
        elif arg == '--time': TIME = True
        

def print_usage_err():
    print(USAGE)
    sys.exit(1)


if __name__ == '__main__':
    main()

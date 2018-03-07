import sys
from parser import parse_data as parse



USAGE = "USAGE: python3 Classifiers.py [--classifier=(str)] [--dataset=(str)] [--trials=(int)] [--train=(float)] [--compare]"

# Arguments to classifier program
CLASSIFIER = 'random-forest'
DATASET = 'iris'
N_TRIALS = 1
PCT_TRAIN = 0.90
COMPARE = False
VERBOSE = None


CLASSIFIERS = ['random-forest', 'decision-tree']
DATASETS = ['iris', 'cars']


def main():
    global CLASSIFIER, N_TRIALS, PCT_TRAIN, VERBOSE
    get_args()

    if VERBOSE is None: VERBOSE = not COMPARE

    random_forest_tests = 0
    random_forest_correct = 0

    decision_tree_tests = 0
    decision_tree_correct = 0


    # If comparing (using both classifiers), or using the random forest only
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
        
        # Run the random forests classifier N_TRIALS times on data from DATASET
        for i in range(N_TRIALS):
            print("\nRunning random forest trial "+str(i+1)+" of "+str(N_TRIALS)+"...")
            correct, tests = run_random_forest(data, pct_train=PCT_TRAIN, verbose=VERBOSE)
            random_forest_correct += correct
            random_forest_tests += tests



    if COMPARE or CLASSIFIER == 'decision-tree':
        if DATASET == 'iris':
            try:
                from decisionTree import run_classifier as run_decision_tree
            except:
                print("Cannot import decision tree classifier for the iris dataset. Exiting.")
                sys.exit()


        elif DATASET == 'cars':
            print("Decision tree not yet implemented for car dataset. Exiting.")
            sys.exit()

        # Run the iris classifier N_TRIALS times
        for i in range(N_TRIALS):
            print("\nRunning decision tree trial "+str(i+1)+" of "+str(N_TRIALS)+"...")
            correct, tests = run_decision_tree(PCT_TRAIN, verbose=VERBOSE)
            decision_tree_correct += correct
            decision_tree_tests += tests

        

    # If we ran the random forest classifier, print the results to the console
    if random_forest_tests != 0:
        print("\nFINAL RESULT FOR RANDOM FOREST CLASSIFIER:")
        print("\t"+str(random_forest_correct)+"/"+str(random_forest_tests)+" were correctly predicted ("\
                  +str(100*random_forest_correct/random_forest_tests)+"%)")


    # If we ran the decision tree classifier, print the results to the console
    if decision_tree_tests != 0:
        print("\nFINAL RESULT FOR DECISION TREE CLASSIFIER:")
        print("\t"+str(decision_tree_correct)+"/"+str(decision_tree_tests)+" were correctly predicted ("\
                  +str(100*decision_tree_correct/decision_tree_tests)+"%)")


    if COMPARE:
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




def get_args():
    '''Parse any command line arguments to the program
        parameters:
            None
        returns:
            None (stored in globals)
    '''

    global CLASSIFIER, N_TRIALS, PCT_TRAIN, DATASET, COMPARE, VERBOSE

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
        

def print_usage_err():
    print(USAGE)
    sys.exit(1)


if __name__ == '__main__':
    main()
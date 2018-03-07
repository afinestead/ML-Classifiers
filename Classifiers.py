import sys
from parser import parse_data as parse


USAGE = "USAGE: python3 Classifiers.py [--classifier=(str)] [--dataset=(str)] [--trials=(int)] [--train=(float)]"

# Arguments to classifier program
CLASSIFIER = 'random-forest'
DATASET = 'iris'
N_TRIALS = 1
PCT_TRAIN = 0.90


CLASSIFIERS = ['random-forest', 'decision-tree']
DATASETS = ['iris', 'cars']

TESTS = 0
TOTAL_CORRECT = 0


def main():
    global CLASSIFIER, N_TRIALS, PCT_TRAIN, TESTS, TOTAL_CORRECT
    get_args()
    print("DATASET="+str(DATASET))

    if CLASSIFIER == 'random-forest':
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
            print("\nRunning trial "+str(i+1)+" of "+str(N_TRIALS)+"...")
            correct, tests = run_random_forest(data, pct_train=PCT_TRAIN)
            TOTAL_CORRECT += correct
            TESTS += tests


    elif CLASSIFIER == 'decision-tree':
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


        # Run the iris classifier N_TRIALS times
        for i in range(N_TRIALS):
            print("\nRunning trial "+str(i+1)+" of "+str(N_TRIALS)+"...")
            correct, tests = run_decision_tree(PCT_TRAIN)
            TOTAL_CORRECT += correct
            TESTS += tests


    print("\nFINAL RESULT: "+str(TOTAL_CORRECT)+"/"+str(TESTS)+" were correctly predicted ("+str(100*TOTAL_CORRECT/TESTS)+"%)")



def get_args():
    '''Parse any command line arguments to the program
        No parameters, no return (stored in globals)'''
    global CLASSIFIER, N_TRIALS, PCT_TRAIN, DATASET

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
        

def print_usage_err():
    print(USAGE)
    sys.exit(1)


if __name__ == '__main__':
    main()
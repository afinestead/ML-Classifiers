PROJECT:
Machine Learning Classifiers

A set of machine learning classifiers designed to cluster data based on similar features
Classifiers include decision trees and random forests.

USAGE:

python3 Classifiers.py [--classifier=(str)] [--dataset=(str)] [--train=(float)] [--trials=(int)]

The classifier argument is optional and can be given as either 'random-forest' or 'decision-tree'. The defualt is 'random-forest'.
	This controls the classifier that is applied to the data.

The dataset argument is optional and can be given as 'iris'. The default is 'iris'.
	This controls the data to be classified.

The train argument is optional and can be any value between 0 and 1, inclusive. The default is 0.90.
	This controls the percent of data from the original dataset that is used to train the model.
	The percent of data that the model is tested on is therefore (1-train).

The trials argument is optional and can be any integer value greater than 0. The default is 1.
	This controls the number of trials that are performed on the classifier and dataset.
	Each trial is independent (the model is re-trained at the start of every trial).


CONTRIBUTORS:
Alex Finestead, University of Washington
Pinzhu Qian, University of Washington



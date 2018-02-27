#!/usr/bin/python3
'''Milestone_A_who_and_what.py
This runnable file will provide a representation of
answers to key questions about your project in CSE 415.

'''

# DO NOT EDIT THE BOILERPLATE PART OF THIS FILE HERE:

CATEGORIES=['Baroque Chess Agent','Feature-Based Reinforcement Learning for the Rubik Cube Puzzle',\
  'Supervised Learning: Comparing Trainable Classifiers']

class Partner():
  def __init__(self, lastname, firstname, uwnetid):
    self.uwnetid=uwnetid
    self.lastname=lastname
    self.firstname=firstname

  def __lt__(self, other):
    return (self.lastname+","+self.firstname).__lt__(other.lastname+","+other.firstname)

  def __str__(self):
    return self.lastname+", "+self.firstname+" ("+self.uwnetid+")"

class Who_and_what():
  def __init__(self, team, option, title, approach, workload_distribution, references):
    self.team=team
    self.option=option
    self.title=title
    self.approach = approach
    self.workload_distribution = workload_distribution
    self.references = references

  def report(self):
    rpt = 80*"#"+"\n"
    rpt += '''The Who and What for This Submission

Final Project in CSE 415, University of Washington, Winter, 2018
Milestone A

Team: 
'''
    team_sorted = sorted(self.team)
    # Note that the partner whose name comes first alphabetically
    # must do the turn-in.
    # The other partner(s) should NOT turn anything in.
    rpt += "    "+ str(team_sorted[0])+" (the partner who must turn in all files in Catalyst)\n"
    for p in team_sorted[1:]:
      rpt += "    "+str(p) + " (partner who should NOT turn anything in)\n\n"

    rpt += "Option: "+str(self.option)+"\n\n"
    rpt += "Title: "+self.title + "\n\n"
    rpt += "Approach: "+self.approach + "\n\n"
    rpt += "Workload Distribution: "+self.workload_distribution+"\n\n"
    rpt += "References: \n"
    for i in range(len(self.references)):
      rpt += "  Ref. "+str(i+1)+": "+self.references[i] + "\n"

    rpt += "\n\nThe information here indicates that the following file will need\n"+\
     "to be submitted (in addition to code and possible data files):\n"
    rpt += "    "+\
     {'1':"Baroque_Chess_Agent_Report",'2':"Rubik_Cube_Solver_Report",\
      '3':"Trainable_Classifiers_Report"}\
        [self.option]+".pdf\n"

    rpt += "\n"+80*"#"+"\n"
    return rpt

# END OF BOILERPLATE.

# Change the following to represent your own information:

alex = Partner("Finestead", "Alexander", "acf8")
pinzhu = Partner("Qian", "Pinzhu", "pinzhu")
team = [alex, pinzhu]

OPTION = '3'
# Legal options are 1, 2, and 3.

title = "Comparing trainable classifiers"
 # In this case, the Python file for the formulation would be named End_Poverty.py.

approach = '''We will first understand the procedure for this assignment. We will then choose two datasets
which we will use for training and classification. We will implement the decision tree and random
forest to classify our data. We plan on using 90 percent of each dataset to train our classifier, and then
use it to classify the remaining 10 percent. Once we have trained our models and classified our data,
we will compare the two classifiers based on performance.'''



workload_distribution = '''Pinzhu will be responsible for implementing the decision tree classifier. Alex will
be responsible for implementing the random forest. Both partners will be responsible for 
comparing classifiers and writing the report. Additionally, both partners will be involved in
selecting the datasets for classification.'''



reference1 = '''Sci-kit documentation for random forests;
    URL: http://scikit-learn.org/stable/modules/ensemble.html'''

reference2 = '''Youtube video on decision tree classifiers;
    URL: https://www.youtube.com/watch?v=LDRbO9a6XPU'''

reference3 = '''Naive Bayes classifier;
    URL: https://www.analyticsvidhya.com/blog/2017/09/naive-bayes-explained/'''

reference4 = '''BBC datasets;
    URL: http://mlg.ucd.ie/datasets/bbc.html'''

our_submission = Who_and_what(team, OPTION, title, approach, workload_distribution, [reference1, reference2, reference3, reference4])

# You can run this file from the command line by typing:
# python3 who_and_what.py

# Running this file by itself should produce a report that seems correct to you.
if __name__ == '__main__':
  print(our_submission.report())
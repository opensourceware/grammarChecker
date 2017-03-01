This is a simple and efficient agggregate bigram model which detects correct grammatical structure. If you are confused between two possible grammatical structures for a sentence, you can use this model to choose the correct grammatical sequence for you. The aggregate bigram models grammatical strucure in the hidden classes between two words. This model successfully learns an approximate form of english grammatical rules. I have also put up a python notebook with theoretical details about training and experimental results.

TRAINING
This application uses Expectation-Maximization algorithm for training. For approximately 20 iterations, this algorithm takes around 5 mins to train.

RUN
Command line arguments:
-i (defaults to 20) number of training iterations
-c (defaults to 5) number of hidden classes which represent grammatical structure

Simply run the code using:
python grammarChecker.py --iterations 20 --class_size 10

You will be asked to enter input sentences on prompt.
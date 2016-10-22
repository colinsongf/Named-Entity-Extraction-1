import numpy as np
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.externals import joblib


features_array_train = []
labels_array_train = []
features_array_test = []
labels_true = []

def parseTrainingData(training_data):
    for line in training_data:
        line_split = line.split(' ')
        label = line_split[1]
        # Create features vectors array
        f_array = [int(i) for i in (line_split[2].rstrip('\n')).split(',')]
        features_array_train.append(f_array)
        # Create labels array
        labels_array_train.append([label])

def parseTestingData(testing_data):
    for line in testing_data:
        line_split = line.split(' ')
        f_array = [int(i) for i in (line_split[2].rstrip('\n')).split(',')]
        features_array_test.append(f_array)
        labels_true.append(line_split[1])

def trainAndPredict(X_train, y_train_text, X_test, target_labels, testing_data):
    lb = preprocessing.LabelBinarizer()
    print "BINARIZING DATA"
    y_train = lb.fit_transform(y_train_text)
    print "DONE"

    print "LOADING MODEL"
    model_clone = joblib.load('my_model.pkl')
    print "PREDICTING ON TEST DATA"
    pred = model_clone.predict(X_test)
    labels_predicted = lb.inverse_transform(pred)

    print 'Precision: ', precision_score(labels_true, labels_predicted, average='weighted')
    print 'Recall: ', recall_score(labels_true, labels_predicted, average='weighted')
    print 'F-Score ', f1_score(labels_true, labels_predicted, average='weighted')

if __name__ == '__main__':
    training_data = open('training_data.txt', 'r')
    testing_data = open('testing_data.txt', 'r')

    parseTrainingData(training_data)
    print "DONE PARSING TRAINING DATA"
    parseTestingData(testing_data)
    print "DONE PARSING TESTING DATA"

    X_train = np.array(features_array_train)
    y_train_text = np.array(labels_array_train)

    X_test = np.array(features_array_test)
    target_labels = ['O', 'B-GPE', 'I-GPE', 'B-ORG', 'I-ORG', 'B-PER', 'I-PER']

    print "TRAINING"
    trainAndPredict(X_train, y_train_text, X_test, target_labels, testing_data)
    print "DONE"

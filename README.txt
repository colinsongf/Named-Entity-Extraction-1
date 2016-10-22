Mukul Surajiwale
Natural Language Processing Fall 2016
Assignment 3

I used a SVM based approach.

---OverView---
generateFeatures.py
   Converts the raw data provided for the assignment into the data that contains
   the feature vectors. It writes this data to the files called "training_data.txt"
   and "testing_data.txt". This process does take about 5 minutes because the
   size of the training data.

   Data Format
      <word>   <label>  <f1, f2, f3, ...fn>

classify.py
   This script will train a multiclassification SVM based on the features vectors
   produced by the generateFeatures.py script.
   WARNING: This process take ~ 4 hours on the full dataset. Thus, I have provided
            an already trained model for you :)

testModel.py
   This script will use the pretrained model that I have provided and run it against
   the testing data. This script with take ~ 4 minutes to run.

   How to run:
      cd <something>/HW3
      enter in:
         python testModel.py

RESULTS:
   Precision: 0.895677352058
   Recall: 0.889040131425
   F-Score: 0.884262642444

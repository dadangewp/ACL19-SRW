
# -*- coding: utf-8 -*-

from dataReader import parse_training
from dataReader import parse_training_cast
from dataReader import parse_testing
from dataReader import parse_testing_cast
import configFeature as cfgFeature
import featureManager
from sklearn import svm
from sklearn import tree
from sklearn import metrics
from sklearn.metrics.scorer import make_scorer
from sklearn.metrics import f1_score, classification_report, accuracy_score, make_scorer
from sklearn.model_selection import cross_val_score, cross_val_predict



DIR_TRAIN = "es_ibereval_training.txt"
DIR_TEST = "es_ibereval_testing.txt"

originalclass = []
predictedclass = []

def classification_report_with_accuracy_score(y_true, y_pred):
    #print (classification_report(y_true, y_pred)) # print classification report
    originalclass.extend(y_true)
    predictedclass.extend(y_pred)
    return accuracy_score(y_true, y_pred) # return accuracy score

if __name__ == '__main__':
    
    print ("started ...")
    feature_manager=featureManager.make_feature_manager()
    dataTrain, dataLabel = parse_testing(DIR_TRAIN)
    dataTest, labelTest = parse_testing(DIR_TEST)
    print ("Training data read") 
    feature_names=cfgFeature.feature_list['feature_names']
    X_train, X_test = feature_manager.create_feature_space(dataTrain, dataTest, feature_names)
    clf = svm.LinearSVC()
    #clf = svm.SVC()
    clf.fit(X_train,dataLabel)
    predicted = clf.predict(X_test)
    #print (len(predicted))
    score_pos = metrics.f1_score(labelTest, predicted, pos_label=1)
    score_neg = metrics.f1_score(labelTest, predicted, pos_label=0)
    acc = metrics.accuracy_score(labelTest, predicted)
    precision = metrics.precision_score(labelTest, predicted, pos_label=1)
    recall = metrics.recall_score(labelTest, predicted, pos_label=1)
    #tn, fp, fn, tp = confusion_matrix(labelTest,predicted).ravel()
    avg = (score_pos + score_neg)/2
    print(acc)
    print(score_pos)
    print(score_neg)
    print(precision)
    print(recall)
    print(avg)
            
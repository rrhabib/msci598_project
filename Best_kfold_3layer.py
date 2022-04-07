import sys
import numpy as np
import sklearn
import pandas as pd
import csv
import nltk

from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding
#nltk.download("punkt")
#nltk.download("wordnet")
nltk.download("omw-1.4")
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm
from sklearn import linear_model
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from feature_engineering import refuting_features, polarity_features, hand_features, gen_or_load_feats
from feature_engineering import word_overlap_features
from feature_engineering import cos_similarity
from utils.dataset import DataSet
from utils.generate_test_splits import kfold_split, get_stances_for_folds
from utils.score import report_score, LABELS, score_submission

from utils.system import parse_params, check_version



def generate_features(stances,dataset,name):
    h, b, y = [],[],[]

    for stance in stances:
        y.append(LABELS.index(stance['Stance']))
        h.append(stance['Headline'])
        b.append(dataset.articles[stance['Body ID']])

    X_overlap = gen_or_load_feats(word_overlap_features, h, b, "features/overlap."+name+".npy")
    X_refuting = gen_or_load_feats(refuting_features, h, b, "features/refuting."+name+".npy")
    X_polarity = gen_or_load_feats(polarity_features, h, b, "features/polarity."+name+".npy")
    X_hand = gen_or_load_feats(hand_features, h, b, "features/hand."+name+".npy")
    cos_sim = cos_similarity(h,b)
    X = np.c_[X_hand, X_polarity, X_refuting, X_overlap, cos_sim]
    return X,y

if __name__ == "__main__":
    check_version()
    parse_params()

    #Load the training dataset and generate folds
    d = DataSet()
    folds,hold_out = kfold_split(d,n_folds=10)
    fold_stances, hold_out_stances = get_stances_for_folds(d,folds,hold_out)

    # Load the competition dataset
    competition_dataset = DataSet("competition_test")
    X_competition, y_competition = generate_features(competition_dataset.stances, competition_dataset, "competition")

    Xs = dict()
    ys = dict()

    # Load/Precompute all features now
    X_holdout,y_holdout = generate_features(hold_out_stances,d,"holdout")
    for fold in fold_stances:
        Xs[fold],ys[fold] = generate_features(fold_stances[fold],d,str(fold))


    best_score = 0
    best_fold = None


    # Classifier for each fold
    for fold in fold_stances:
        ids = list(range(len(folds)))
        del ids[fold]


        X_train_list = [Xs[i] for i in ids]
        y_train_list = [ys[i] for i in ids]

        X_train = np.vstack(tuple([Xs[i] for i in ids]))

        y_related = []
        for k in y_train_list:
            current = []
            y_related.append(current)
            for stance in k:
                if stance == 3:
                    current.append(3)
                else:
                    current.append(2)
        y_train = np.hstack(tuple(y_related))

        X_test = Xs[fold]
        y_test = ys[fold]

        # Breakdown of  classification problem into three stages was inspired by the Project Below
        # Wu, X., Cheng, S., & Chai, Z. (2017). Fake news stance detection - CS229.STANFORD.EDU. Stanford. Retrieved March 2022, from https://cs229.stanford.edu/proj2017/final-reports/5244160.pdf

        #Model 1 - logistic regression to classigy related and unrelated 
        clf1 = linear_model.LogisticRegression(max_iter=500)

        clf1.fit(X_train, y_train)
        x_rel_unrel = []
        y_rel_unrel = []
        y_train = np.hstack(tuple(y_train_list))
        for index, k in enumerate(list(y_train)):
            if k != 3:
                y_rel_unrel.append(k)
                x_rel_unrel.append(X_train[index])

        x_rel_unrel = np.array(x_rel_unrel)
        y_rel_unrel = np.array(y_rel_unrel)

        # Model 2: Classify Having (Agree or Disagree) or Neural
        clf2 = linear_model.LogisticRegression(max_iter=500)
        clf2.fit(x_rel_unrel, y_rel_unrel)

        x_agree_disagree = []
        y_agree_disagree = []
        y_train = np.hstack(tuple(y_train_list))
        for index, k in enumerate(list(y_train)):
            if k != 2 and k != 3:
                y_agree_disagree.append(k)
                x_agree_disagree.append(X_train[index])

        x_agree_disagree = np.array(x_agree_disagree)
        y_agree_disagree = np.array(y_agree_disagree)


        ###test incorporating tifdf
        #X_tfidf = trainTFIDF(x_rel_unrel, 1, 1)


        # Model 3: Classify Agree & Disagree
        clf3 = Sequential()
        clf3.add(Dense(200, activation='relu'))
        #clf3.add(Embedding(1000, 100, input_length=44))
        #clf3.add(Embedding(1000, 100, input_length=45))
        #clf3.add(LSTM(128))
        clf3.add(Dense(4, activation='softmax'))
        clf3.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        clf3.fit(x_rel_unrel, y_rel_unrel, epochs=5, batch_size=128)
        #clf3.fit(X_tfidf, y_rel_unrel, epochs=1, batch_size=128)

        #Neural Net Predictions
        nn_results1 = clf3.predict(X_test)
        predictions1 = []
        for j in nn_results1:
            index_max = max(range(len(j)), key=j.__getitem__)
            predictions1.append(index_max)
        print(predictions1[1])
        print("Neural Net Complete")
        # prediction in the different stages
        results = clf1.predict(X_test)
        for index, k in enumerate(list(results)):
            if k != 3:
                results[index] = clf2.predict(np.array(X_test[index]).reshape(1, -1))
                if results[index] != 2:
                    results[index] = predictions1[index]


                    #predictions = []
                    #for j in test:
                     #   index_max = max(range(len(j)), key=j.__getitem__)
                     #   predictions.append(index_max)
                    #results[index] = predictions[index]

        predicted = [LABELS[int(a)] for a in results]
        actual = [LABELS[int(a)] for a in y_test]

        fold_score, _ = score_submission(actual, predicted)
        max_fold_score, _ = score_submission(actual, actual)

        score = fold_score/max_fold_score

        print("Score for fold "+ str(fold) + " was - " + str(score))
        if score > best_score:
            best_score = score
            best_clf1 = clf1
            best_clf2 = clf2
            best_clf3 = clf3


    #Run on Holdout set and report the final score on the holdout set

    # Neural Net Predictions
    nn_results2 = best_clf3.predict(X_holdout)
    predictions2 = []
    for j in nn_results2:
        index_max = max(range(len(j)), key=j.__getitem__)
        predictions2.append(index_max)
    print(predictions2[1])
    print("Neural Net Complete")
    holdout_results = best_clf1.predict(X_holdout)
    for index, k in enumerate(list(holdout_results)):
        if k != 3:
            holdout_results[index] = best_clf2.predict(np.array(X_holdout[index]).reshape(1, -1))
            if holdout_results[index] != 2:
                holdout_results[index] = predictions2[index]
                #predictions = []
                #for j in test:
                 #   index_max = max(range(len(j)), key=j.__getitem__)
                #    predictions.append(index_max)
               # holdout_results[index] = predictions.reshape(1, -1)

    predicted = [LABELS[int(a)] for a in holdout_results]
    actual = [LABELS[int(a)] for a in y_holdout]

    print("Scores on the dev set")
    report_score(actual,predicted)
    print("")
    print("")

    #Run on competition dataset
    # Neural Net Predictions
    nn_results3 = best_clf3.predict(X_competition)
    predictions3 = []
    for j in nn_results3:
        index_max = max(range(len(j)), key=j.__getitem__)
        predictions3.append(index_max)
    print(predictions3[1])
    print("Neural Net Complete")

    comp_results = best_clf1.predict(X_competition)
    for index, k in enumerate(list(comp_results)):
        if k != 3:
            comp_results[index] = best_clf2.predict(np.array(X_competition[index]).reshape(1, -1))
            if comp_results[index] != 2:
                comp_results[index] = predictions3[index]

    predicted = [LABELS[int(a)] for a in comp_results]
    actual = [LABELS[int(a)] for a in y_competition]

    print("Scores on the test set")
    report_score(actual,predicted)

    #print(predicted[:10])

    #predictNp = np.array(predicted)
    #predictNp = predictNp.transpose()
    #np.savetxt('answers.csv', [predictNp], fmt='%s')

    #unlabelledFile = open('fnc-1/competition_test_stances_unlabeled.csv','r', newline='')
    #unlabelled = unlabelledFile.read()

    content = []
    with open('fnc-1/competition_test_stances_unlabeled.csv', 'r', encoding="utf8") as file:
        reader = csv.reader(file)
        for row in reader:
            #print(row)
            content.append(row)

    #print('predict size ', len(predicted))
    #print(content[0])
    #print(content[1])
    #print (len(content))
    predicted.insert(0,'Stance')
    #print(predicted[:10])

    for i in range(0,len(content)):
            content[i].append(predicted[i])

    print(len(content), ' x ' , len(content[0]))

    out_file = open("answer.csv", 'w', newline='', encoding="utf8")
    with out_file:
        writer = csv.writer(out_file)
        writer.writerows(content)


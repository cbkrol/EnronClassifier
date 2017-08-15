def test_classifier(clf, dataset, feature_list, folds = 1000, conf_matrix_filename="confusion_matrix.png", title="Confusion Matrix"):
  
    from sklearn.cross_validation import StratifiedShuffleSplit
    from featureFormat import featureFormat
    from targetFeatureSplit import targetFeatureSplit
    import numpy as np
    import matplotlib.pyplot as plt
    import itertools


    PERF_FORMAT_STRING = "\
    \nAccuracy: {:>0.{display_precision}f}\nPrecision: {:>0.{display_precision}f}\n\
    Recall: {:>0.{display_precision}f}\nF1: {:>0.{display_precision}f}\nF2: {:>0.{display_precision}f}"
    RESULTS_FORMAT_STRING = "\nTotal predictions: {:4d}\nTrue positives: {:4d}\nFalse positives: {:4d}\
    \nFalse negatives: {:4d}\nTrue negatives: {:4d}"
    
    
    data = featureFormat(dataset, feature_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)
    cv = StratifiedShuffleSplit(labels, folds, random_state = 42)
    true_negatives = 0
    false_negatives = 0
    true_positives = 0
    false_positives = 0
    for train_idx, test_idx in cv: 
        features_train = []
        features_test  = []
        labels_train   = []
        labels_test    = []
        for ii in train_idx:
            features_train.append( features[ii] )
            labels_train.append( labels[ii] )
        for jj in test_idx:
            features_test.append( features[jj] )
            labels_test.append( labels[jj] )
        
        ### fit the classifier using training set, and test on test set
        clf.fit(features_train, labels_train)
        predictions = clf.predict(features_test)
        for prediction, truth in zip(predictions, labels_test):
            if prediction == 0 and truth == 0:
                true_negatives += 1
            elif prediction == 0 and truth == 1:
                false_negatives += 1
            elif prediction == 1 and truth == 0:
                false_positives += 1
            elif prediction == 1 and truth == 1:
                true_positives += 1
            else:
                print ("Warning: Found a predicted label not == 0 or 1.")
                print ("All predictions should take value 0 or 1.")
                print ("Evaluating performance for processed predictions:")
                break
    try:
        total_predictions = true_negatives + false_negatives + false_positives + true_positives
        accuracy = 1.0*(true_positives + true_negatives)/total_predictions
        precision = 1.0*true_positives/(true_positives+false_positives)
        recall = 1.0*true_positives/(true_positives+false_negatives)
        f1 = 2.0 * true_positives/(2*true_positives + false_positives+false_negatives)
        f2 = (1+2.0*2.0) * precision*recall/(4*precision + recall)
        print (PERF_FORMAT_STRING.format(accuracy, precision, recall, f1, f2, display_precision = 5))
        print (RESULTS_FORMAT_STRING.format(total_predictions, true_positives, false_positives, false_negatives, true_negatives))
        print ("")

        class_names = ["POI", "not POI"]

        # Confusion matrix
        conf_arr = np.array([[true_positives,false_positives], 
                [false_negatives,true_negatives]])

        print("confusion array: ", conf_arr)
        plt.imshow(conf_arr, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)

        fmt =  'd'
        thresh = conf_arr.max() / 2.
        for i, j in itertools.product(range(conf_arr.shape[0]), range(conf_arr.shape[1])):
            plt.text(j, i, format(conf_arr[i, j], fmt),
                    horizontalalignment="center",
                    color="white" if conf_arr[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

        plt.savefig(conf_matrix_filename)
        print("saved figure to filename: ", conf_matrix_filename)
        plt.show()


        #---------------------------------------
        #fig = plt.figure()
        #plt.clf()
        #ax = fig.add_subplot(111)     
        #ax.set_aspect(1)
        #res = ax.imshow(np.array(norm_conf), cmap=plt.cm.jet, 
        #        interpolation='nearest')   
        #width, height = conf_arr.shape              
        #for x in range(width):
        #    for y in range(height):
        #        print("x and y: ", x, y)
        #        ax.annotate(str(conf_arr[x][y]), xy=(y, x), 
        #                   horizontalalignment='center',
        #                   verticalalignment='center')
        #cb = fig.colorbar(res)
        #tick_marks = np.arange(len(class_names))
        #plt.xticks(tick_marks, class_names, rotation=45)
        #plt.yticks(tick_marks, class_names)
        #plt.savefig('confusion_matrix.png', format='png')
        #plt.show()


    except:
        print ("Got a divide by zero when trying out:", clf)
        print ("Precision or recall may be undefined due to a lack of true positive predictions.")


def evaluate(clf, dataset, feature_list, features, labels, params):
    """
    Function used to evaluate our algorithm -- prints out the mean precision, recall, and accuracy.
    :param clf: Classifier algorithm (e.g. LogisticRegression(), DecisionTreeClassifier()
    :param features:
    :param labels: Feature we're trying to classify -- POI / non-POI
    :param params: Parameters used in the classifier pipeline.
                    e.g. {
                        "clf__criterion": ["gini", "entropy"],
                        "clf__min_samples_split": [10,15,20,25]
                    }
    :return: Prints the accuracy, precision, and recall score.
    """

    from sklearn.cross_validation import train_test_split
    from sklearn.grid_search import GridSearchCV



    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.3, random_state=42)



    precision_values = []
    recall_values = []
    accuracy_values = []
    print (clf)
    
    grid = GridSearchCV(clf, params)
    grid.fit(features_train, labels_train)
    print("********* grid *******: ", grid)
    print("**********************")
    
    # print(getattr(grid, 'grid_scores_', None))
    
    # summarize the results of the grid search
    print("-----> best score: ", grid.best_score_)
    print("-----> best estimator: ", grid.best_estimator_)
    print("-----> best params: ", grid.best_params_)

    
    clf = grid.best_estimator_
    test_classifier(clf, dataset, feature_list)








def test_random_data_splits(clf, dataset, feature_list, features, labels, params, trials):
    """
    Function used to evaluate our algorithm -- prints out the mean precision, recall, and accuracy.
    :param clf: Classifier algorithm (e.g. LogisticRegression(), DecisionTreeClassifier()
    :param features:
    :param labels: Feature we're trying to classify -- POI / non-POI
    :param params: Parameters used in the classifier pipeline.
                    e.g. {
                        "clf__criterion": ["gini", "entropy"],
                        "clf__min_samples_split": [10,15,20,25]
                    }
    :return: Prints the accuracy, precision, and recall score.
    """

    from sklearn.cross_validation import train_test_split
    from sklearn.grid_search import GridSearchCV
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from statistics import mean


    precision_values = []
    recall_values = []
    accuracy_values = []
    f1_values = []

    print (clf)
    for i in range(0, trials):

        features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.3, random_state=42)
        print("train_test_split iteration:", i)

        grid = GridSearchCV(clf, params, 'recall')
        grid.fit(features_train, labels_train)
        print("********* grid recall scoring *******: ", grid)
        print("**********************")

        # summarize the results of the grid search

        print("-----> iteration best score: ", grid.best_score_)
        print("-----> iteration best estimator: ", grid.best_estimator_)
        print("-----> iteration best params: ", grid.best_params_)

    
        # print(getattr(grid, 'grid_scores_', None))
        clf = grid.best_estimator_
        pred = clf.predict(features_test)
        precision_values.append(precision_score(labels_test, pred))
        recall_values.append(recall_score(labels_test, pred))
        accuracy_values.append(accuracy_score(labels_test, pred))
        f1_values.append(f1_score(labels_test, pred))
        #print(f1_values)


   

    print ('Recall score mean: ', mean(recall_values))
    print ('Precision score mean: ', mean(precision_values))
    print ('Accuracy score mean: ' , mean(accuracy_values))
    print ('F1 score mean: ' , mean(f1_values))

    print ("==============================================================")







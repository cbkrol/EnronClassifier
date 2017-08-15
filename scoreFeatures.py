def scoreTopFeatures(data_dict, feature_list, num_features ):
    """
    Function for selecting our KBest features.
    :param data_dict: List of employees and features
    :param feature_list: List of features to select
    :param num_features: Number (k) of features to select in the algorithm (k = 11)
    :return: Returns a list of the KBest feature names
    """

    import sys
    sys.path.append("./myPy3Tools/")
    
    # myPy3Tools
    from featureFormat import featureFormat
    from targetFeatureSplit import targetFeatureSplit

    from sklearn.feature_selection import SelectKBest


    data = featureFormat(data_dict, feature_list)
    target, features = targetFeatureSplit(data)

    clf = SelectKBest(k = num_features)
    clf = clf.fit(features, target)


    # lambda effectively creates an inline function. For example, you can rewrite this example:
    #   max(gs_clf.grid_scores_, key=lambda x: x[1])
    
    # Using a named function:
    # def element_1(x):
    #     return x[1]
    # max(gs_clf.grid_scores_, key=element_1)
    
    # In this case, max() will return the element in that array whose 
    # second element (x[1]) is larger than all of the other elements' second elements. 
    # Another way of phrasing it is as the function call implies: return the max element, using x[1] as the key.
    feature_weights = {}
   
    for idx, feature_score in enumerate(clf.scores_):
        # print("feature:  ", feature_list[1:][idx], "score: ", feature_score)
        feature_weights[feature_list[1:][idx]] = feature_score
        
    best_features = sorted(feature_weights.items(), key = lambda k: k[1], reverse = True)[:num_features]
    new_features = []
    for k, v in best_features:
        new_features.append(k)
        print("feature: ", k, "  score: ",v)

    return new_features
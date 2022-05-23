from __future__ import division, print_function
from __future__ import print_function
from __future__ import division
from sklearn.metrics import roc_auc_score
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
from sklearn.pipeline import Pipeline

np.random.seed(123)


def get_link_feats(links, source_embeddings, target_embeddings, operator, index_mapping_nodeID):
    """Compute link features for a list of pairs"""
    features = []
    index_mapping_nodeID=index_mapping_nodeID.tolist()
    for l in links:
        a=index_mapping_nodeID.index(l[0])# nodeid映射位置
        b=index_mapping_nodeID.index(l[1])
        f = operator(source_embeddings[a], target_embeddings[b])
        features.append(f)
    return features


def evaluate_link_prediction(train_pos, train_neg, test_pos, test_neg, source_embeds, target_embeds, operators, operators_name, index_mapping_nodeID):
    test_results = defaultdict(lambda: [])
    test_pred_true = defaultdict(lambda: [])

    for id,operator in enumerate(operators):
        train_pos_feats = np.array(get_link_feats(train_pos, source_embeds, target_embeds, operator,index_mapping_nodeID))
        train_neg_feats = np.array(get_link_feats(train_neg, source_embeds, target_embeds, operator,index_mapping_nodeID))
        test_pos_feats = np.array(get_link_feats(test_pos, source_embeds, target_embeds, operator,index_mapping_nodeID))
        test_neg_feats = np.array(get_link_feats(test_neg, source_embeds, target_embeds, operator,index_mapping_nodeID))

        train_pos_labels = np.array([1] * len(train_pos_feats))
        train_neg_labels = np.array([-1] * len(train_neg_feats))

        test_pos_labels = np.array([1] * len(test_pos_feats))
        test_neg_labels = np.array([-1] * len(test_neg_feats))
        train_data = np.vstack((train_pos_feats, train_neg_feats))
        train_labels = np.append(train_pos_labels, train_neg_labels)

        test_data = np.vstack((test_pos_feats, test_neg_feats))
        test_labels = np.append(test_pos_labels, test_neg_labels)

        logistic = linear_model.LogisticRegression()
        logistic.fit(train_data, train_labels)
        test_predict = logistic.predict_proba(test_data)[:, 1] #预测概率大小
        test_roc_score = roc_auc_score(test_labels, test_predict)
        test_results[operators_name[id]].extend([test_roc_score, test_roc_score])

    return test_results, test_pred_true
    
def link_examples_to_features(link_examples, transform_node, operator):
    return [
        operator(transform_node(src), transform_node(dst)) for src, dst in link_examples
    ]

def link_prediction_classifier(max_iter=2000):
    lr_clf = linear_model.LogisticRegressionCV(Cs=10, cv=10, scoring="roc_auc", max_iter=max_iter)
    return Pipeline(steps=[("sc", StandardScaler()), ("clf", lr_clf)])

def evaluate_roc_auc(clf, link_features, link_labels):
    predicted = clf.predict_proba(link_features)

    # check which class corresponds to positive links
    positive_column = list(clf.classes_).index(1)
    return roc_auc_score(link_labels, predicted[:, positive_column])


def labelled_links(positive_examples, negative_examples):
    return (
        positive_examples + negative_examples,
        np.repeat([1, 0], [len(positive_examples), len(negative_examples)]),
    )
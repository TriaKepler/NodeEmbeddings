from numpy.random import choice as rnd_choice
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

def edges_spliter(graph, p=0.1, max_trial=100):
    edges = [(vi, vj) for vi, vj in graph.edges]
    nodes_num = len(graph.nodes)
    pos_edges_num = int(p * len(edges))
    if nodes_num ** 2 < 2 * pos_edges_num:
        raise Exception('No sufficient number of negative edges')
    labels = np.zeros(2 * pos_edges_num)
    pos_edges = np.asarray(edges)[rnd_choice(len(edges), pos_edges_num)]
    neg_edges = []

    for i in range(pos_edges_num):
        n1 = rnd_choice(nodes_num)
        for trail in range(max_trial):
            n2 = rnd_choice(nodes_num)
            if n2 not in graph.neighbors(n1):
                break
        if n2 in graph.neighbors(n1):
            raise Exception("Max iteration reached")
        else:
            neg_edges.append((n1, n2))

    labels[: labels.size // 2] = 1
    neg_edges = np.array(neg_edges)
    examples = np.concatenate((pos_edges, neg_edges), axis=0)
    graph.remove_edges_from(pos_edges)
    return graph, examples, labels

def train_test_graph_for_link_prediction(graph, train_split_ratio=0.75, random_state=123):
    test_graph, test_examples, test_labels = edges_spliter(graph)
    train_graph, train_examples, train_labels = edges_spliter(test_graph)
    (train_train_examples,
    train_test_examples,
    train_train_labels,
    train_test_labels) = train_test_split(train_examples, train_labels, train_size=train_split_ratio, test_size=1 - train_split_ratio, random_state=random_state)
    return [test_graph, train_graph], [test_examples, test_labels], [train_train_examples, train_train_labels], [train_test_examples, train_test_labels]

def link_examples_to_features(link_examples, embeddings, binary_operator):
    return [binary_operator(embeddings[src], embeddings[dst]) for src, dst in link_examples]

def train_link_prediction_model(
    link_examples, link_labels, get_embedding, binary_operator):
    clf = link_prediction_classifier()
    link_features = link_examples_to_features(
        link_examples, get_embedding, binary_operator
    )
    clf.fit(link_features, link_labels)
    return clf

def link_prediction_classifier(max_iter=2000):
    lr_clf = LogisticRegressionCV(Cs=10, cv=10, scoring="roc_auc", max_iter=max_iter)
    return Pipeline(steps=[("sc", StandardScaler()), ("clf", lr_clf)])

def evaluate_link_prediction_model(
    clf, link_examples_test, link_labels_test, get_embedding, binary_operator):
    link_features_test = link_examples_to_features(
        link_examples_test, get_embedding, binary_operator)
    score = evaluate_roc_auc(clf, link_features_test, link_labels_test)
    return score

def evaluate_roc_auc(clf, link_features, link_labels):
    predicted = clf.predict_proba(link_features)
    positive_column = list(clf.classes_).index(1)
    return roc_auc_score(link_labels, predicted[:, positive_column])

def run_link_prediction(binary_operator, examples, labels, examples_model_selection, labels_model_selection, embeddings):
    clf = train_link_prediction_model(
        examples, labels, embeddings, binary_operator
    )
    score = evaluate_link_prediction_model(
        clf,
        examples_model_selection,
        labels_model_selection,
        embeddings,
        binary_operator,
    )

    return {
        "classifier": clf,
        "binary_operator": binary_operator,
        "score": score,
    }

def train_evaluate_link_prediction(test_examples_labels, train_train_examples_labels, train_test_examples_labels, embeddings_train, embeddings_test):
    operator_hadamard = lambda u, v: u * v
    operator_hadamard.__name__ = 'hadamard'
    operator_l1 = lambda u, v: np.abs(u - v)
    operator_l1.__name__ = 'l1'
    operator_l2 = lambda u, v: (u - v) ** 2
    operator_l2.__name__ = 'l2'
    operator_avg = lambda u, v: (u + v) / 2.0
    operator_avg.__name__ = 'avg'

    binary_operators = [operator_hadamard, operator_l1, operator_l2, operator_avg]
    results = [run_link_prediction(op, *train_train_examples_labels, *train_test_examples_labels, embeddings_train) for op in binary_operators]
    best_result = max(results, key=lambda result: result["score"])

    print(f"Best result from operator '{best_result['binary_operator'].__name__}'")

    print("Results over training set")
    print(pd.DataFrame(
        [(result["binary_operator"].__name__, result["score"]) for result in results],
        columns=("name", "ROC AUC score"),
    ).set_index("name"))

    for idx, op in enumerate(binary_operators):
        test_score = evaluate_link_prediction_model(
        results[idx]["classifier"],
        *test_examples_labels,
        embeddings_test,
        op)
        print(f"ROC AUC score on test set using '{op.__name__}': {test_score}")

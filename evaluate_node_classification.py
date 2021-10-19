from numpy import mean
from numpy import std
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import RepeatedKFold
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical

def get_model(n_inputs, n_outputs):
    model = Sequential([Dense(64, activation='relu', input_dim=n_inputs), \
    Dense(64, activation='relu'), \
    Dense(n_outputs, activation='softmax')])
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model

def evaluate_model(X, y, classes_num, epochs=100, splits_num=10, repeats_num=3, seed=123):
    results = []
    print(X.shape, y.shape)
    n_inputs, n_outputs = X.shape[1], classes_num
    cv = RepeatedKFold(n_splits=splits_num, n_repeats=repeats_num, random_state=seed)
    for train_ix, test_ix in cv.split(X):
        X_train, X_test = X[train_ix], X[test_ix]
        y_train, y_test = y[train_ix], y[test_ix]
        model = get_model(n_inputs, n_outputs)
        model.fit(X_train, to_categorical(y_train, classes_num), verbose=0, epochs=epochs)
        yhat = model.predict(X_test)
        yhat = np.argmax(yhat, axis=1)
        acc = accuracy_score(y_test, yhat)
        print(f'{acc}')
        results.append(acc)
    return mean(results), std(results)

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras import losses, optimizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import datasets
import numpy as np
import time

start = time.time()

def oneHot(y):
    """
    :param y: array, vector
    :return: 2D sparse matrix, consisting of 0 and 1
    """
    lyst=[i for i in range(10)]
    y_bool = [lyst==i for i in y]
    return np.array(y_bool, dtype=np.int)


digits = datasets.load_digits()
X = digits.data
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    random_state=2333)
y_train_t = oneHot(y_train)

model = Sequential()
model.add(Dense(128, input_dim=len(X[0])))
model.add(Activation("sigmoid"))
model.add(Dense(10))
model.add(Activation('softmax'))
model.compile(optimizer=optimizers.Adam(),loss=losses.mean_squared_error)

model.fit(X_train,
          y_train_t,
          epochs=50,
          batch_size=100)
y_predict = model.predict_classes(X_test)
print("ACC on test:", accuracy_score(y_predict, y_test))
print("ACC on train:", accuracy_score(y_predict, y_test))

end = time.time()

print("Running time: ", end-start)



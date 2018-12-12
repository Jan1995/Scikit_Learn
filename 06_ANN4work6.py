from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras import losses, optimizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

np.random.seed(2333)
X = np.random.normal(0, 3, size=(400, 2))
y = np.array(X[:,0]**2 + X[:,1]**2 < 10.5, dtype=np.int)

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    random_state=233)

model = Sequential()
model.add(Dense(50, input_dim=len(X[0])))
model.add(Activation("sigmoid"))
model.add(Dense(2))
model.add(Activation('softmax'))
model.compile(optimizer=optimizers.Adam(),loss=losses.mean_squared_error)
model.fit(X_train,
          np.array([[1,0] if i==0 else [0,1] for i in y_train]),
          epochs=500,
          batch_size=50
          )
y_predict = model.predict_classes(X_test)
print("ACC on test:", accuracy_score(y_predict, y_test))
print("ACC on train:", accuracy_score(y_predict, y_test))

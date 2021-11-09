import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn import datasets
from keras.utils.np_utils import to_categorical

def plot_decision_boundary(X, model):
    x_span = np.linspace(min(X[:, 0])-0.25, max(X[:, 0])+0.25, 50)
    y_span = np.linspace(min(X[:, 1])-0.25, max(X[:, 1])+0.25, 50)
    xx, yy = np.meshgrid(x_span, y_span)
    xx_, yy_ = xx.ravel(), yy.ravel()
    grid = np.c_[xx_, yy_]
    pred_func = np.argmax(model.predict(grid), axis=-1)
    z = pred_func.reshape(xx.shape)
    plt.contourf(xx, yy, z)

n_pts = 500
centres = [[-1, 1], [-1, -1], [1, -1], [1, 1], [0, 0]]
X, labels = datasets.make_blobs(n_samples = n_pts, random_state=123, centers=centres, cluster_std=0.4)
# plt.scatter(X[labels==0, 0], X[labels==0, 1])
# plt.scatter(X[labels==1, 0], X[labels==1, 1])
# plt.scatter(X[labels==2, 0], X[labels==2, 1])
# plt.scatter(X[labels==3, 0], X[labels==3, 1])
# plt.scatter(X[labels==4, 0], X[labels==4, 1])
# plt.show()
# One hot encode the labels
labels_cat = to_categorical(labels, 5)
#print(labels_cat)
model = Sequential()
model.add(Dense(units=5, input_shape=(2,), activation='softmax'))
model.compile(Adam(learning_rate=0.1), loss='categorical_crossentropy', metrics=['accuracy'])
print(X.shape)
model.fit(x=X, y=labels_cat, verbose=1, batch_size=50, epochs=100)
plot_decision_boundary(X, model)
plt.scatter(X[labels==0, 0], X[labels==0, 1])
plt.scatter(X[labels==1, 0], X[labels==1, 1])
plt.scatter(X[labels==2, 0], X[labels==2, 1])
plt.scatter(X[labels==3, 0], X[labels==3, 1])
plt.scatter(X[labels==4, 0], X[labels==4, 1])

x = -1
y = -1
point = np.array([[x,y]])
prediction = np.argmax(model.predict(point), axis=-1)
plt.plot([x], [y], marker='x', markersize = 10, color='r')
print("Prediction: ", prediction)
plt.show()








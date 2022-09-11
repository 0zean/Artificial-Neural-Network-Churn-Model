#Importing Libraries
import os

os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

import warnings

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import plot_model
from sklearn.metrics import confusion_matrix

warnings.filterwarnings('ignore')
tf.random.set_seed(123)
####### DATA PREPROCESSING ########

# Importing Dataset
dataset = pd.read_csv("Churn_Modelling.csv")
x = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values


# Encoding categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_x_1 = LabelEncoder()
x[:, 1] = labelencoder_x_1.fit_transform(x[:, 1])
labelencoder_x_2 = LabelEncoder()
x[:, 2] = labelencoder_x_2.fit_transform(x[:, 2])
ct = ColumnTransformer([("Country", OneHotEncoder(), [1])], remainder = 'passthrough')
x = ct.fit_transform(x)

# Splitting the dataset into training set and test set
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    test_size = 0.2,
                                                    random_state = 0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

######## ANN #########
class acc_stop(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy')>0.995):
            print("\nReached 99.5% accuracy, cancelling training")
            self.model.stop_training = True

classifier = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation=tf.nn.relu, input_dim=x_train.shape[1]),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(256, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(256, activation=tf.nn.relu),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

classifier.summary()
plot_model(classifier, 'model.png', show_shapes=True)

classifier.compile(loss='binary_crossentropy',
                   optimizer='adam',
                   metrics=['accuracy'])

#rlr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.05, patience=20, verbose=1)
es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

# Fit ANN to training set
history = classifier.fit(x=x_train, y=y_train,
                         batch_size=32,
                         validation_data=(x_test, y_test),
                         verbose=1,
                         epochs=200,
                         callbacks=[acc_stop(), es])

tf.keras.backend.clear_session()

######## PREDICTIONS & EVALUATION ##########

# Predicting the test set results
y_pred = classifier.predict(x_test)
y_pred = (y_pred > 0.5)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

sn.heatmap(cm, annot=True)
plt.show()


# Plot history of training metrics
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()

plt.show()
plt.savefig('Training_validation_accuracy.png')

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
plt.savefig('Training_validation_loss.png')

# Predicting a single new observation

""" Geography: France
Credit Score: 600
Gender: Male
Age: 40
Tenure: 3
Balance: 60000
Number of Products: 2
Has Credit Card: Yes
Is Active Member: Yes
Estimated Salary: 50000"""

new_prediction = classifier.predict(sc.transform(np.array([[1.0, 0.0, 0.0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
new_prediction = (new_prediction > 0.5)

######## Evaluating, Improving, Tuning #########

# Evaluating the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score


def build_classifier():
    classifier = tf.keras.Sequential()
    classifier.add(tf.keras.layers.Dense(units = 128, kernel_initializer = 'uniform', activation = 'relu', input_dim = 12))
    classifier.add(tf.keras.layers.Dense(units = 256, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(tf.keras.layers.Dense(units = 256, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(tf.keras.layers.Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier_CV = KerasClassifier(build_fn = build_classifier, epochs = 100, batch_size = 10)

accuracies = cross_val_score(estimator=classifier_CV, X = x_train, y= y_train, cv=10)
mean = accuracies.mean()
variance = accuracies.std()


# Tuning the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV


def build_classifier(optimizer):
    classifier = tf.keras.Sequential()
    classifier.add(tf.keras.layers.Dense(units = 128, kernel_initializer = 'uniform', activation = 'relu', input_dim = 12))
    classifier.add(tf.keras.layers.Dense(units = 256, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(tf.keras.layers.Dense(units = 256, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(tf.keras.layers.Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier_GS = KerasClassifier(build_fn = build_classifier)

parameters = {'batch_size': [25, 32],
              'epochs': [100, 500],
              'optimizer': ['adam', 'rmsprop']}

grid_search = GridSearchCV(estimator = classifier_GS,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)

grid_search = grid_search.fit(x_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_

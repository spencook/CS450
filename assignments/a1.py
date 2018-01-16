from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

iris = datasets.load_iris()

# show the dataset
print(iris.data)

# show the target values
print(iris.target)

# show the target names
print(iris.target_names)

# randomize and split into training and testing set
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3)

# create and train model
classifier = GaussianNB()
model = classifier.fit(x_train, y_train)

# make predictions
targets_predicted = model.predict(x_test)
print(targets_predicted)

print(100*(accuracy_score(y_test, targets_predicted)))

# step 5
class HardCodedClassifier:
    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
        return self

    def predict(self, x_test):
        return [0 for x in range(len(x_test))]

classifier = HardCodedClassifier()
model = classifier.fit(x_train, y_train)
targets_predicted = model.predict(x_test)

targets_predicted = model.predict(x_test)
print(targets_predicted)

print(100*(accuracy_score(y_test, targets_predicted)))

# above and beyond

data_csv = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data")
print(data_csv)
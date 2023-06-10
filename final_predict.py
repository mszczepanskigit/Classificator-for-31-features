from sklearn.svm import SVC
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import csv

data_train = np.genfromtxt('data_train.csv', delimiter=';')

predict_data = np.genfromtxt('data_test_public_no_classes.csv', delimiter=';')

selected_columns = [1] + list(range(3, 34))
data = data_train[:, selected_columns]
predict = predict_data[:, selected_columns]
num_records = data.shape[0]

class_ = data_train[:, -1]
classes = [0, 1]

multi_results = np.zeros((71, 30))+2

for iters in range(10):
    classifier = KNeighborsClassifier(n_neighbors=2)
    indices = np.random.choice(num_records, size=int(0.8 * num_records), replace=False)
    remaining_indices = np.setdiff1d(np.arange(num_records), indices)
    data_80 = data[indices]
    class_80 = class_[indices]
    data_20 = data[remaining_indices]
    class_20 = class_[remaining_indices]

    classifier.fit(data_80, class_80)
    Y_pred = classifier.predict(predict)
    multi_results[:, iters] = Y_pred

for iters in range(10, 30):
    classifier = SVC(C=2.3, kernel='rbf', gamma=0.15)
    indices = np.random.choice(num_records, size=int(0.8 * num_records), replace=False)
    remaining_indices = np.setdiff1d(np.arange(num_records), indices)
    data_80 = data[indices]
    class_80 = class_[indices]
    data_20 = data[remaining_indices]
    class_20 = class_[remaining_indices]

    classifier.fit(data_80, class_80)
    Y_pred = classifier.predict(predict)
    multi_results[:, iters] = Y_pred

for i in range(multi_results.shape[0]):
    multi_results[i, :] = np.round(np.mean(multi_results[i, :]))

result = multi_results[:, 0]

with open('final.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=';')

    for i, value in enumerate(result):
        writer.writerow([i, int(value)])



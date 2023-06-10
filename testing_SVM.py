from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt

data_train = np.genfromtxt('data_train.csv', delimiter=';')

selected_columns = [1] + list(range(3, 34))
data = data_train[:, selected_columns]
num_records = data.shape[0]

class_ = data_train[:, -1]
classes = [0, 1]

# Ustaw parametry SVM
kernel_type = 'rbf'

to_plot = []

for gam in [0.15]:
    for C_value in [2.3]:
        results = []
        for iters in range(100):
            classifier = SVC(C=C_value, kernel=kernel_type, gamma=gam)
            indices = np.random.choice(num_records, size=int(0.9 * num_records), replace=False)
            remaining_indices = np.setdiff1d(np.arange(num_records), indices)
            data_80 = data[indices]
            class_80 = class_[indices]
            data_20 = data[remaining_indices]
            class_20 = class_[remaining_indices]

            # Wytrenuj klasyfikator SVM
            classifier.fit(data_80, class_80)

            # Wykonaj predykcję na danych 20%
            Y_pred = classifier.predict(data_20)

            # Oblicz wynik i dodaj do listy wyników
            res = np.sum(np.abs(np.array(Y_pred) - np.array(class_20))) / len(Y_pred)
            results.append(res)

        to_plot.append(np.mean(np.array(results)))
        print(
            f"Final average result for {np.round(C_value, 1)} and gamma {np.round(gam, 2)} is: {np.mean(np.array(results))}")


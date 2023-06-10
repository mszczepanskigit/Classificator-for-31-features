from sklearn.neighbors import KNeighborsClassifier

import numpy as np

#np.random.seed(666)

data_train = np.genfromtxt('data_train.csv', delimiter=';')

selected_columns = [1] + list(range(3, 34))
data = data_train[:, selected_columns]
num_records = data.shape[0]

class_ = data_train[:, -1]
classes = [0, 1]

for neig in range(1, 21):
    results = []
    for iters in range(200):
        classifier = KNeighborsClassifier(n_neighbors=neig)
        indices = np.random.choice(num_records, size=int(0.8 * num_records), replace=False)
        remaining_indices = np.setdiff1d(np.arange(num_records), indices)
        data_80 = data[indices]
        class_80 = class_[indices]
        data_20 = data[remaining_indices]
        class_20 = class_[remaining_indices]

        classifier.fit(data_80, class_80)
        Y_pred = classifier.predict(data_20)

        res = np.sum(np.abs(np.array(Y_pred) - np.array(class_20)))/len(Y_pred)
        results.append(res)

    print(f"Final average result for {neig} neighbors is: {np.mean(np.array(results))}")

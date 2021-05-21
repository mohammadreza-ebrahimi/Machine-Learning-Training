import matplotlib as mlp
import matplotlib.pyplot as plt
import numpy as np
# %%
from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784', version=1)
mnist.keys()
# Data : Contains an array with one row per instance and one column per feature.
# target : labeled data
# %%
X, y = mnist['data'], mnist['target']
X.shape, y.shape
# There are 70,000 images, each image has 784 features (28 * 28)px,
# so each image represents one pixel's intensity from 0 to 255.
# %%
import pandas as pd

data_df = pd.DataFrame(X)
data_df, data_df.describe()
data_df.loc[3]
# %%
mnist_frame, mnist_cat, mnist_fn, mnist_tn, mnist_descr, mnist_dt, mnist_url = mnist['frame'], mnist['categories'], \
                                                                               mnist['feature_names'], \
                                                                               mnist['target_names'], \
                                                                               mnist['DESCR'], mnist['details'], \
                                                                               mnist['url']
mnist_cat, mnist_fn, mnist_tn
mnist_dt
# %%
some_digit = X[0]
some_digit
# %%
some_digit_image = some_digit.reshape(28, 28)
# plt.imshow(some_digit_image)  # RGB
plt.imshow(some_digit_image, cmap=mlp.cm.binary, interpolation='nearest')
plt.axis('off')
plt.show()
# %%
y[0]
# %%
y = y.astype(np.uint8)
y[0], y
# %%
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
# %%
# Training binary classifier
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)
y_train, y_train_5

# %%
from sklearn.linear_model import SGDClassifier

sgd_cls = SGDClassifier(random_state=42)
sgd_cls.fit(X_train, y_train_5)
# sgd_cls.fit(X_train, y_train) #Wrong model training
# %%
sgd_cls.predict([X_train[0]]), sgd_cls.predict([X[1]])
# %%
y_train[1]
# %%
from sklearn.model_selection import cross_val_score

scores_cls = cross_val_score(sgd_cls, X_train, y_train_5, scoring="accuracy")  # Default "cv" is 5


def display_score(scores):
    print('Scores', scores)
    print('Mean of Accuracy', scores.mean())
    print('Standard Deviation', scores.std())


display_score(scores_cls)

# %%

plt.hist(X_train)
plt.show()
# %%
plt.hist(y, histtype='bar', bins=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], ec='black', rwidth=0.5,
         align='left')
plt.xticks(np.arange(0, 10, 1))
plt.show()

# %%
plt.figure(figsize=(15, 10))
for i in range(30):
    X_train_image = X_train[i].reshape(28, 28)
    plt.subplot(6, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(X_train_image, cmap=mlp.cm.binary)
    # plt.xlabel(y_train[i])
    plt.xlabel(y_train_5[i])
plt.show()
# %%
y_train_5[1]
# %%
plt.figure(figsize=(15, 10))
for i in range(30):
    X_train_image = X_train[i].reshape(28, 28)
    plt.subplot(6, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(X_train_image, cmap=mlp.cm.binary)
    # plt.xlabel(y_train[i])
    plt.xlabel(sgd_cls.predict([X_train[i]]))
plt.show()
# %%

from sklearn.model_selection import cross_val_predict

y_train_pred = cross_val_predict(sgd_cls, X_train, y_train_5, cv=5)

# %%
from sklearn.metrics import confusion_matrix

confusion_matrix(y_train_pred, y_train_5)
# %%
53115 + 916 + 1464 + 4505  # TN + FP + FN + TP
# %% Recall = TP / (TP + FN) (Total targets) ; Precision = TP / (TP + FP) (Total target predicted)
from sklearn.metrics import precision_score, recall_score

precision_score(y_train_5, y_train_pred), recall_score(y_train_5, y_train_pred)
# %%
from sklearn.metrics import f1_score

f1_score(y_train_5, y_train_pred)
# %%
y_scores = sgd_cls.decision_function([X_train[0]])
y_scores
# %%
thresholds = 0
y_pred = (y_scores > thresholds)
y_pred
# %%
thresholds = 8000
y_pred = (y_scores < thresholds)
# y_pred = (y_scores > thresholds)
y_pred
# %%
y_scores = cross_val_predict(sgd_cls, X_train, y_train_5, cv=5,
                             method="decision_function")
# %%
from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)
precisions, recalls, thresholds


# %%
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
    plt.legend(loc="center right", fontsize=16)
    plt.xlabel("Threshold", fontsize=16)
    plt.grid(True)
    plt.axis([-50000, 50000, 0, 1])


recall_90_precision = recalls[np.argmax(precisions >= 0.90)]
threshold_90_precision = thresholds[np.argmax(precisions >= 0.90)]

plt.figure(figsize=(8, 4))
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.plot([threshold_90_precision, threshold_90_precision], [0., 0.9], "r:")
plt.plot([-50000, threshold_90_precision], [0.9, 0.9], "r:")
plt.plot([-50000, threshold_90_precision], [recall_90_precision, recall_90_precision], "r:")
plt.plot([threshold_90_precision], [0.9], "ro")
plt.plot([threshold_90_precision], [recall_90_precision], "ro")
plt.show()


# %%

def plot_precision_vs_recall(precisions, recalls):
    plt.plot(recalls, precisions, "b-", linewidth=2)
    plt.xlabel("Recall", fontsize=16)
    plt.ylabel("Precision", fontsize=16)
    plt.axis([0, 1, 0, 1])
    plt.grid(True)


plt.figure(figsize=(8, 6))
plot_precision_vs_recall(precisions, recalls)
plt.plot([0.658919, 0.658919], [0., 0.9], "r:")
plt.plot([0.0, 0.658919], [0.9, 0.9], "r:")
plt.plot([0.658919], [0.9], "ro")
plt.show()

# %%
np.argmax(precisions >= 0.90)
sgd_cls.decision_function([X_train[55954]])
# %%
threshold_90_precision = thresholds[np.argmax(precisions >= 0.90)]
threshold_90_precision

# %%
y_train_pred_90 = (y_scores >= threshold_90_precision)
y_train_pred_90

# %%
precision_score(y_train_5, y_train_pred_90), recall_score(y_train_5, y_train_pred_90)
# %%  Multiclass classifier
# SGD has OvA (One versus All) classifier as default
sgd_cls.fit(X_train, y_train)  # not y_train_5
sgd_cls.predict([some_digit])

# %%
some_digit_scores = sgd_cls.decision_function([some_digit])
some_digit_scores
# %% To transform score to the target value we have
np.argmax(some_digit_scores)

# %%
some_digit1 = X_train[1]
y_train[1]
# %%
some_digit1_scores = sgd_cls.decision_function([some_digit1])
sgd_cls.predict([some_digit1])
some_digit1_scores, np.argmax(some_digit1_scores)
# %%
sgd_cls.classes_
# %% for training model with OvO
from sklearn.multiclass import OneVsOneClassifier

ovo_clf = OneVsOneClassifier(SGDClassifier(random_state=42))  # Constructing as instance n(n-1)/n
ovo_clf.fit(X_train, y_train)  # training the model
ovo_clf.predict([some_digit])
# %%
ovo_clf.predict([X_train[2]]), y_train[2]
# %% Random forest in classification
from sklearn.ensemble import RandomForestClassifier

forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)
forest_clf.fit(X_train, y_train)
forest_clf.predict([some_digit])
# %%
forest_clf.predict_proba([some_digit])
# %%
from sklearn.model_selection import cross_val_score

cross_val_score(sgd_cls, X_train, y_train, scoring='accuracy')
# %%
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scale = scaler.fit_transform(X_train.astype(np.float64))
cross_val_score(sgd_cls, X_train_scale, y_train, cv=3, scoring='accuracy')
# %%
y_train_pred = cross_val_predict(sgd_cls, X_train, y_train)
conf_mx = confusion_matrix(X_train, y_train_pred)
conf_mx

# %%
plt.matshow(conf_mx, cmap=plt.cm.gray)
# %%
row_sums = conf_mx.sum(axis=1, keepdims=True)
norm_conf_mx = conf_mx / row_sums

# %%
np.fill_diagonal(norm_conf_mx, 0)
plt.matshow(norm_conf_mx, cmap=plt.cm.gray)


# %%
def plot_digits(instances, images_per_row=10, **options):
    size = 28
    images_per_row = min(len(instances), images_per_row)
    images = [instance.reshape(size, size) for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((size, size * n_empty)))
    for row in range(n_rows):
        rimages = images[row * images_per_row: (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    image = np.concatenate(row_images, axis=0)
    plt.imshow(image, cmap=mlp.cm.binary, **options)
    plt.axis("off")
    plt.show()


# %%
cl_a, cl_b = 3, 5
X_aa = X_train[(y_train == cl_a) & (y_train_pred == cl_a)]
X_ab = X_train[(y_train == cl_a) & (y_train_pred == cl_b)]
X_ba = X_train[(y_train == cl_b) & (y_train_pred == cl_a)]
X_bb = X_train[(y_train == cl_b) & (y_train_pred == cl_b)]
plt.figure(figsize=(8, 8))
plt.subplot(221);
plot_digits(X_aa[:25], images_per_row=5)
plt.subplot(222);
plot_digits(X_ab[:25], images_per_row=5)
plt.subplot(223);
plot_digits(X_ba[:25], images_per_row=5)
plt.subplot(224);
plot_digits(X_bb[:25], images_per_row=5)

# %%

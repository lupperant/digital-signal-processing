import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import pandas as pd
import seaborn as sns
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# Resize images to
SIZE = 128
# Capture images and labels into arrays.
# Start by creating empty lists.
data = pd.DataFrame()
labels = []
columns = []
# for directory_path in glob.glob("cell_images/train/*"):
#for wheat
for directory_path in glob.glob("D:/pythonProject/images/train/*"):
#for rice
#for directory_path in glob.glob("D:/pythonProject/rice/*"):
    for img_path in glob.glob(os.path.join(directory_path, "*.xlsx")):
        excel_data = pd.read_excel(img_path)
        label = directory_path.split("\\")[-1]
        data = pd.concat([data, excel_data], ignore_index=True)
        k = excel_data.shape
        columns.append(label)
        for i in range(k[0]):
            labels.append(label)
labels.pop(0)
print(len(labels))
print(columns)
df = pd.DataFrame(data)

df.to_csv('your_array.csv', header=False, index=False)
df = pd.read_csv('your_array.csv')
a = df.values
train_images = np.array(a)
train_labels = np.array(labels)

x_train, x_test, y_train, y_test = train_test_split(train_images, train_labels, test_size=0.12)
x_train = np.nan_to_num(x_train)
x_test = np.nan_to_num(x_test)

# Extract features from training images
image_features = x_train
X_for_ML = image_features
test_features = x_test
#
# Создание модели
clf = ExtraTreesClassifier(max_depth=25)

# Обучение модели на обучающей выборке
clf.fit(x_train, y_train)
clf.fit(x_train, y_train)
# Предсказание классов для тестовой выборки
test_prediction = clf.predict(test_features)
# # Inverse le transform to get original label back.

# Оценка качества модели
accuracy = accuracy_score(y_test, test_prediction)
print("Accuracy:", accuracy)
print(classification_report(y_test, test_prediction))

cm = confusion_matrix(y_test, test_prediction)
fig, ax = plt.subplots()  # Sample figsize in inches
sns.set(font_scale=1.6)
sns.heatmap(cm, annot=True, linewidths=.5, ax=ax)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()
plt.close()
# Check results on a few random images
import random

n = random.randint(0, x_test.shape[0] - 1)  # Select the index of image to be loaded for testing
img = x_test[n]
# # Extract features and reshape to right dimensions
input_img = np.expand_dims(img, axis=0)  # Expand dims so the input is (num images, x, y, c)

input_img_for_RF = np.reshape(img, (input_img.shape[0], -1))
# # # Predict
img_prediction = clf.predict(input_img_for_RF)

print("The prediction for this image is: ", img_prediction)
print("The actual label for this image is: ", y_train[n])
print(img_prediction == y_train[n])

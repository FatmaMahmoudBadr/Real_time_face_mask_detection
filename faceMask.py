import os.path
import pandas as pd
from skimage.feature import hog
from skimage.transform import resize
from skimage.io import imread
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

def image_hog_features(image_path):
    image = imread(image_path)
    resized_image = resize(image, (64, 64))
    observation = hog(resized_image, orientations=9, pixels_per_cell=(8, 8),
                      cells_per_block=(2, 2), visualize=False, channel_axis=2)

    return observation

def visualize_hog_features(image_path):
  image = imread(image_path)
  resized_image = resize(image, (64, 64))

  fd, observation = hog(resized_image, orientations=9, pixels_per_cell=(2, 2),
                          cells_per_block=(2, 2), visualize=True, channel_axis=2)


  plt.figure(figsize=(13,13))

  ax = plt.subplot(1, 2, 1)
  plt.imshow(resized_image, cmap='gray')
  plt.title("Original Image")
  plt.axis("off")

  ax = plt.subplot(1, 2, 2)
  plt.imshow(observation, cmap='gray')
  plt.title("Hog Image")
  plt.axis("off")

  plt.show()

def load_data(annotation_folder_path,images_folder_path):
    df = pd.DataFrame(columns=["xmin", "xmax", "ymin", "ymax", "label"])
    x = []
    for filename in os.listdir(annotation_folder_path):
        if filename.endswith(".xml"):
            filepath = os.path.join(annotation_folder_path, filename)
            tree = ET.parse(filepath)
            root = tree.getroot()
            image_name = root.find('filename').text
            image_path = os.path.join(images_folder_path, image_name)
            features = image_hog_features(image_path)
            objects = root.findall('object')
            for element in objects:
                x.append(features)
                label = element.find('name').text
                position = element.find('bndbox')
                xmin = position.find('xmin').text
                xmax = position.find('xmax').text
                ymin = position.find('ymin').text
                ymax = position.find('ymax').text
                df2 = {'xmin': xmin, 'xmax': xmax, 'ymin': ymin, 'ymax': ymax, 'label': label}
                df = df._append(df2, ignore_index=True)

    features_data = pd.DataFrame(x)
    dataset = pd.concat([features_data, df], axis=1)

    return dataset

def bestParameters(X, Y, classifier, parameter, list):
    x_train2, x_val, y_train2, y_val = train_test_split(X, Y, test_size=0.25, random_state=4)
    train_accuracy_values = []
    val_accuracy_values = []
    for p in list:
        model = classifier(**{parameter: p, "random_state": 3})
        model.fit(x_train2, y_train2)
        y_pred_train = model.predict(x_train2)
        y_pred_val = model.predict(x_val)
        acc_train = accuracy_score(y_train2, y_pred_train)
        acc_val = accuracy_score(y_val, y_pred_val)
        train_accuracy_values.append(acc_train)
        val_accuracy_values.append(acc_val)
    plt.plot(list, train_accuracy_values, label='acc train')
    plt.plot(list, val_accuracy_values, label='acc val')
    plt.legend()
    plt.grid(axis='both')
    plt.xlabel(parameter+' parameter')
    plt.ylabel('accuracy')
    plt.title('Effect of entered parameter on accuracy')
    plt.show()

annotation_path = "D:\semester 6\Machine Learning\Annotations"
images_path = "D:\semester 6\Machine Learning\images"
dataset = load_data(annotation_path, images_path)
visualize_hog_features("D:\semester 6\Machine Learning\images\maksssksksss6.png")

X = dataset.drop('label', axis=1)
X.columns = X.columns.astype(str)
Y = dataset['label']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=4)

scaler = preprocessing.StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

max_depth_values = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
max_iter_value = [50, 100, 130, 150, 180, 200, 250, 300]
bestParameters(X_train, y_train, DecisionTreeClassifier, "max_depth", max_depth_values)
bestParameters(X_train, y_train, LogisticRegression, "max_iter", max_iter_value)

LR_model = LogisticRegression(max_iter=150)
LR_model.fit(X_train, y_train)
LR_pred = LR_model.predict(X_test)
test_accuracy_LR = accuracy_score(y_test, LR_pred)

LRCV_model = LogisticRegressionCV()
LRCV_model.fit(X_train, y_train)
LRCV_pred = LRCV_model.predict(X_test)
test_accuracy_LRCV = accuracy_score(y_test, LRCV_pred)

DT_model = DecisionTreeClassifier(max_depth=5)
DT_model.fit(X_train, y_train)
DT_pred = DT_model.predict(X_test)
DT_test_accuracy = accuracy_score(y_test, DT_pred)

RF_model = RandomForestClassifier()
RF_model.fit(X_train,y_train)
RF_pred = RF_model.predict(X_test)
RF_test_accuracy = accuracy_score(y_test,RF_pred)

print(f"Accuracy of Logistic Regression = {test_accuracy_LR*100} %")
print(f"Accuracy of Logistic Regression with cross validation = {test_accuracy_LRCV*100} %")
print(f"Accuracy of Decision Tree = {DT_test_accuracy*100} %")
print(f"Accuracy of Random Forest = {RF_test_accuracy*100} %")
print(classification_report(y_test, LR_pred))
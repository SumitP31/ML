import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.metrics import classification_report
from bkchod import super_print


# Importing the training Dataset and lables
folder_path = "D:/ML/Assignment/Data/"
file_name = "mushroom_trn_data.csv"
MushroomData = pd.read_csv(folder_path + file_name)
Labels = pd.read_csv(folder_path + "mushroom_trn_class_labels.csv", header=None)

# Replacing ? with null values using replace
MushroomData = MushroomData.replace("?", np.nan)
MushroomData = pd.get_dummies(MushroomData, dtype=int)

# convertign training Labels to numericals equivalent using one hot encoding
Labels.rename(columns={1: "labels"}, inplace=True)
Labels.drop(0, inplace=True, axis=1)
Labels = Labels.replace("e", "1")
Labels = Labels.replace("p", "0")

df = pd.concat([MushroomData, Labels], axis="columns")

# train test split
X = df.iloc[:, 0:-1]
y = df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=64
)

# print function
def sup_print(clf: GridSearchCV, X_test: pd.DataFrame, y_test: pd.DataFrame, name: str) -> None:
    print(f"-------------{name}----------------" + "\n")
    print("The best parameters for " + str(clf.best_estimator_))
    print("Accuracy is :" + str(clf.score(X_test, y_test)))
    print(f"The Classification report of {name}")
    print(classification_report(y_test, clf.predict(X_test)))
    print(
        "\n"
        + "------------------------------------------------------------------------------------------------"
    )


# Classifier classes----------------------------------------------------------------------------------------------------------
class classification:
    def __init__(self, data, X_trn, y_trn, X_tst, y_tst):
        self.df = data
        self.X_train = X_trn
        self.y_train = y_trn
        self.X_test = X_tst
        self.y_test = y_tst

        # Decision Tree Classifier implementation ----------------------------------------------------------------------------

    def dt(self):
        dt = DecisionTreeClassifier()
        dt_para = {
            "criterion": ("gini", "entropy"),
            "max_features": ("sqrt", "log2"),
            "max_depth": (10, 40),
            "ccp_alpha": (0.01, 0.005),
        }
        dt_clf = GridSearchCV(dt, dt_para)
        dt_clf.fit(self.X_train, self.y_train)        
        return dt_clf

    # Random Forest classifier implementation ------------------------------------------------------------------------
    def rf(self):
        name = "Random Forest Classifier"
        rf = RandomForestClassifier()
        rf_para = {
            "n_estimators": (10, 5, 25),
            "criterion": ("gini", "entropy", "log_loss"),
            "random_state": (32, 70, 0),
            "max_features": ("sqrt", "log2"),
        }
        rf_clf = GridSearchCV(rf, rf_para)
        rf_clf.fit(self.X_train, self.y_train)
        return rf_clf

    # K-nearest neighbors implementation------------------------------------------------------------------------------
    def kn(self):
        name = "KNeighbors"
        kn = KNeighborsClassifier()
        kn_para = {
            "n_neighbors": (5, 10, 1, 20),
            "algorithm": ("auto", "ball_tree", "kd_tree", "brute"),
        }
        kn_clf = GridSearchCV(kn, kn_para)
        kn_clf.fit(self.X_train, self.y_train)
        return kn_clf

    # Naive Bayes Gaussian classification----------------------------------------------------------------------------
    def gb(self):
        gb = GaussianNB()
        gb_para = {"var_smoothing": (10e-7, 10e-8, 10e-9, 10e-10)}
        gb_clf = GridSearchCV(gb, gb_para)
        gb_clf.fit(self.X_train, self.y_train)
        return gb_clf


# testing different classifiers and comparing their results
model = classification(df, X_train, y_train, X_test, y_test)
# op = model.dt()
# model.rf()
# model.kn()
# model.gb()
sup_print(model.dt(), X_test, y_test, name="DICK TREE")

sup_print(model.gb(), X_test, y_test, name="RUSSIAN GB ROAD")

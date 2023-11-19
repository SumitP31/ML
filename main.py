import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import RFECV
import matplotlib.pyplot as plt
import warnings


warnings.filterwarnings("ignore")
K_best = SelectKBest(f_classif, k=30)

# Importing the training Dataset and lables--------------------------------------------------------------------------------------------
folder_path = "D:/ML/Assignment/Data/"
file_name = "mushroom_trn_data.csv"
MushroomData = pd.read_csv(folder_path + file_name)
Labels = pd.read_csv(folder_path + "mushroom_trn_class_labels.csv", header=None)
Test = pd.read_csv(folder_path + "mushroom_tst_data.csv")

# Replacing ? with null values using replace--------------------------------------------------------------------------------------------
MushroomData = MushroomData.replace("?", np.nan)
MushroomData = pd.get_dummies(MushroomData, dtype=int)
# for test data--------------------------------------------------------------------------------
Test = Test.replace("?", np.nan)
Test = pd.get_dummies(Test, dtype=int)

# convertign training Labels to numericals equivalent using one hot encoding-----------------------------------------------------------
Labels.rename(columns={1: "labels"}, inplace=True)
Labels.drop(0, inplace=True, axis=1)
Labels = Labels.replace("e", "1")
Labels = Labels.replace("p", "0")

df = pd.concat([MushroomData, Labels], axis="columns")


    
# Classifier classes----------------------------------------------------------------------------------------------------------------------------------
class classification:
    def __init__(self, data, p= 0.8, test_size = 0.3,n_features = 20,rd_st = 0):
        self.df = data
        
        # Train Test split  and Feature Selection
        x = self.df.iloc[:,0:-1]
        y = self.df.iloc[:, -1]
        X = K_best.fit_transform(x, y)
        # featselec = VarianceThreshold(threshold=(p*(1 -p)))
        # X = featselec.fit_transform(x)        
        
       
        X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = test_size, random_state=rd_st)
        
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        
        
    # ADAboost------------------------------------------------------------------------------------------------------------
    def ada(self):
        name = 'AdaBoost Classifier'
        dt = DecisionTreeClassifier()
        rf = RandomForestClassifier()
        gb = GaussianNB()
        mn = MultinomialNB()
        lr = LogisticRegression()
        
        ada = AdaBoostClassifier()
        ada_para = {'estimator': (rf,gb,dt,lr,mn),
            'n_estimators':(10,50,100),
            'random_state':(0,10)
            }
        ada_clf = GridSearchCV(ada,ada_para, scoring = 'f1_macro' )
        ada_clf.fit(self.X_train,self.y_train)
        report(ada_clf, 5, self.X_train, self.y_train, self.X_test, self.y_test,name) 
        return ada_clf
        
        
    # Decision Tree Classifier implementation ----------------------------------------------------------------------------
    def dt(self):
        name = 'Decision Tree Classifier'
        dt = DecisionTreeClassifier()
        dt_para = {
            "criterion": ("gini", "entropy"),
            "max_features": ("sqrt", "log2"),
            "max_depth": (10, 40),
            "ccp_alpha": (0.01, 0.005),
        }
        dt_clf = GridSearchCV(dt, dt_para, scoring = 'f1_macro')      
        dt_clf.fit(self.X_train, self.y_train)      
        report(dt_clf, 5, self.X_train, self.y_train, self.X_test, self.y_test,name) 
        return dt_clf

    # Random Forest classifier implementation ------------------------------------------------------------------------
    def rf(self,min_feat=1,cv = 5):
        name = "Random Forest Classifier"
        rf = RandomForestClassifier()        
        rf_para = {"n_estimators": (10, 5, 25),
            "criterion": ("gini", "entropy", "log_loss"),
            "random_state": (32, 70, 0),
            "max_features": ("sqrt", "log2"),
        }
        rf_clf = GridSearchCV(rf, rf_para, scoring = 'f1_macro')
        rf_clf.fit(self.X_train, self.y_train)
        report(rf_clf, 5, self.X_train, self.y_train, self.X_test, self.y_test,name)
        return rf_clf

    # K-nearest neighbors implementation------------------------------------------------------------------------------
    def kn(self,min_feat=1,cv = 5):
        name = "KNeighbors Classifier"
        kn = KNeighborsClassifier()
        kn_para = {"n_neighbors": (2, 5, 8, 10, 12),
            "algorithm": ("auto", "ball_tree", "kd_tree", "brute"),
            }
        kn_clf = GridSearchCV(kn, kn_para, scoring = 'f1_macro')
        kn_clf.fit(self.X_train, self.y_train)
        self.kn_pred = kn_clf.predict(self.X_test)
        report(kn_clf, 5, self.X_train, self.y_train, self.X_test, self.y_test,name)
        return kn_clf

    # Naive Bayes Gaussian classification----------------------------------------------------------------------------
    def gb(self,min_feat=1,cv = 5):
        name = 'Naive Bayes GaussianNB'
        gb = GaussianNB()
        gb_para = {"var_smoothing": (10e-3, 10e-6, 10e-9, 10e-10)}
        gb_clf = GridSearchCV(gb, gb_para,scoring="f1_macro")
        gb_clf.fit(self.X_train, self.y_train)
        self.gb_pred = gb_clf.predict(self.X_test)
        report(gb_clf, 5, self.X_train, self.y_train, self.X_test, self.y_test,name)
        return gb_clf
    
    # Multinomial Naive Bayes Classifier--------------------------------------------------------------------------------
    def mn(self,min_feat=1,cv = 5):
        name = 'Multinomial Naive Bayes Classifier'
        mn = MultinomialNB()
        mn_para = {"alpha": (1,1e-3,1e-6,1e-9)}
        mn_clf = GridSearchCV(mn,mn_para, scoring = "f1_macro")
        mn_clf.fit(self.X_train,self.y_train)  
        report(mn_clf, 5, self.X_train, self.y_train, self.X_test, self.y_test,name)        
        return mn_clf
    
    # Logistic Regression-------------------------------------------------------------------------------------------------
    def lr(self,min_feat=1, cv = 5):
        name = 'Logistic Regression'
        lr = LogisticRegression()
        lr_para ={"random_state": (0, 20, 50),
            "solver": ('lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga')
        }
        lr_clf = GridSearchCV(lr,lr_para,scoring="f1_macro")
        lr_clf.fit(self.X_train,self.y_train)
        report(lr_clf, 5, self.X_train, self.y_train, self.X_test, self.y_test,name)
    
    # Linear SVM
                    
        
# print function-----------------------------------------------------------------------------------------------------------------------------------------------
def report(clf: GridSearchCV,splits: int, X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.DataFrame, name: str) -> None:
    predVal = clf.predict(X_test) 
    val = cross_validate(clf, X_train, y_train, cv= splits)
    confusionMat = metrics.ConfusionMatrixDisplay(confusion_matrix = metrics.confusion_matrix(y_test, predVal), display_labels = [False, True])
    print(f"-------------{name}----------------" + "\n")
    print("The best parameters for " + str(clf.best_estimator_))
    print("K-fold validation score: " + str(np.mean(val['test_score'])) )
    print("Accuracy is :" + str(clf.score(X_test, y_test)))
    print("\nThe Classification report of "+ name)
    print("\nF-1 score:")
    print("--->micro "+str(f1_score(y_test,predVal,average='micro')))
    print("--->macro "+str(f1_score(y_test,predVal,average='macro')))
    print("--->weigthed "+str(f1_score(y_test,predVal,average='weighted')))
    print("\n" + classification_report(y_test, predVal))    
    print( "\n"+ "------------------------------------------------------------------------------------------------")
    confusionMat.plot()
    plt.title(name)
    plt.show()
    # plt.pause(10)
    # plt.close()

       
# testing different classifiers and comparing their results---------------------------------------------------------------------------------
model1 = classification(df)
# model1.ada()
# model1.dt()
# model1.rf()
# model1.kn()
# model1.gb()
# model1.mn()
# model1.lr()


model2 = classification(df,n_features=25)
# modeel2.ada()
# model2.dt()
# model2.rf()
# model2.kn()
# model2.gb()
# model2.mn()
# model2.lr()




# Saving the predicted labels
k_best = MushroomData.columns[K_best.get_support()]
tst_data = Test[k_best]
predictedLabels = model1.ada().predict(tst_data)
np.savetxt(folder_path+"/test_labels.csv", predictedLabels ,delimiter=',',fmt='%s')


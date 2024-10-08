import glob
import os
from mlxtend.feature_selection import SequentialFeatureSelector as SeqFeatSel
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFECV
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd
from itertools import combinations
import os
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from pandas.api.types import is_numeric_dtype
import warnings
import sys
from sklearn.exceptions import ConvergenceWarning
import collections
from statsmodels.discrete import discrete_model
from datetime import date
import math
from statsmodels.stats.outliers_influence import variance_inflation_factor

sns.set()

class listFiles:
    #i.e. listFiles(path="path")
    def __init__(self, **kwargs): #path, dict, file_id, list
        if kwargs == {}:
            print("please enter the path value")
            exit()
        else:
            if "path" not in kwargs.keys():
                print("please enter the path value")
                exit()
            else:
                path = kwargs["path"]
            if "dict" not in kwargs.keys():
                self.dict = {}
            else:
                self.dict = kwargs["dict"]
            if "list" not in kwargs.keys():
                self.list = []
            else:
                self.list = kwargs["list"]
            if "file_id" not in kwargs.keys():
                file_id = 0
            else:
                file_id = kwargs["file_id"]

        for file in glob.glob(f"{path}/*"):
            if os.path.isdir(file):
                subdir = file
                listFiles(path=subdir, dict=self.dict, file_id=file_id, list=self.list)
            else:
                if os.path.isfile(file):
                    if ".txt" in file:
                        self.list.append(file)
                        directory = path + "/"
                        filename = file.split(directory)[1]
                        if filename not in self.dict.keys():
                            self.dict[filename] = file_id
                            file_id += 1


class filePreparation:
    def __init__(self, filename="", workingDir="", cdrDir="", label="Pass"):
        files = listFiles(path=cdrDir)
        file_dict = files.dict
        self.data = pd.read_csv(f"{workingDir}/{filename}")
        self.data = self.data.drop(["Unnamed: 0"], axis=1,
                         errors="ignore")  # , "Source Alias", "Destination Alias", "Disconnect Reason", "Day", "Time", "Source IP"],axis=1)  # Source Alias", "Destination Alias", "Disconnect Reason", "Day", "Time", "Source IP"], axis = 1)
        self.data = self.data.dropna(axis=0)
        if label in self.data.columns:
            self.data[label] = self.data[label].map({"y": 1, "n": 0}).reset_index(drop=True)
        self.data["Source IP Location"] = self.data["Source IP Location"].map({"Internal": 0, "External": 1})
        self.data["CollaborationEdge"] = self.data["CollaborationEdge"].apply(normalization)
        self.data["Cloud"] = self.data["Cloud"].apply(normalization)

        columns = self.data.columns.tolist()
        # Perform one-hot encoding and drop one category to avoid multicollinearity
        oneHotOF = pd.get_dummies(self.data["Original File"], prefix='category', drop_first=True)
        oneHotDK = pd.get_dummies(self.data["Disconnect Key"], prefix='category', drop_first=True)
        # Concatenate the one-hot encoded columns with the original DataFrame
        oneHotOFColumns = oneHotOF.columns.tolist()
        oneHotDKColumns = oneHotDK.columns.tolist()
        if label in columns:
            estimators = columns[0:-1] + oneHotDKColumns + oneHotOFColumns
            estimators.append(label)
        else:
            estimators = columns + oneHotDKColumns + oneHotOFColumns
        estimators.remove("Original File")
        estimators.remove("Disconnect Key")
        #estimators.remove("Unnamed: 0")
        combinedData = pd.concat([self.data, oneHotDK, oneHotOF], axis=1)
        # Drop the original category column
        combinedData = combinedData.drop("Original File", axis=1)
        combinedData = combinedData.drop("Disconnect Key", axis=1)
        combinedData = combinedData[estimators]
        combinedData = combinedData*1
        if label in columns:
            self.estimators = estimators[0:-1]
        else:
            self.estimators = estimators
        #self.data["Original File"] = data["Original File"].map(file_dict)
        #self.data["Day and time (sec)"] = self.data["Day and time (sec)"].apply(dayAndTimeNorm)
        #self.data["Day and time (sec)"].to_csv("dayandtime.csv")
        self.data = combinedData
        self.data = self.data * 1
        self.numericData = self.data.select_dtypes(include=['number', 'bool'])
        self.numericEstimators = self.numericData.columns.tolist()[0:-1] # except the label
        #input(f"Check if {filename} has any variable with null values")

    def actualValues(self, label):
        return self.data[label]


class regressionModel:
    def __init__(self, model): #def __init__(self, **kwargs):
        self.model = model

    def trainTestGen(self,file="", trainFile="", testFile="", random_state=999999, estimators=[]):
        if not file.empty:
            self.singleFile = file
            self.trainFile = ""
            self.testFile = ""
            all_columns = self.singleFile.columns.values.tolist()
            X_data = self.singleFile[all_columns[0:-1]]
            Y_data = self.singleFile[all_columns[-1]]
            if random_state != 999999:
                self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X_data, Y_data,
                                                                                    test_size=0.2,
                                                                                    random_state=random_state)
            else:
                self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X_data, Y_data,
                                                                                   test_size=0.2)
        else:
            if "trainFile" !="" and "testFile" != "":
                self.trainFile = trainFile
                self.testFile = testFile
                self.singleFile = ""
                missingCols = set(self.trainFile) - set(self.testFile)
                for col in missingCols:
                    self.testFile[col] = 0
                all_columns = self.trainFile.columns.values.tolist()
                self.X_train = self.trainFile[all_columns[0:-1]]
                self.Y_train = self.trainFile[all_columns[-1]]
                self.X_test = self.testFile[all_columns[0:-1]]
                self.Y_test = self.testFile[all_columns[-1]]
            else:
                print("one or more files are missing")
                exit()

        if estimators != []:
            # X_train type is Pandas
            print("estimators in trainTestGen are ", estimators)
            X_train_selected = self.X_train[estimators]
            X_test_selected = self.X_test[estimators]
            self.estimators = estimators
            scaler = StandardScaler()
            self.X_train_scaled = scaler.fit_transform(X_train_selected)
            scaler_test = StandardScaler()
            self.X_test_scaled = scaler_test.fit_transform(X_test_selected)
        else:
            print("estimators missing")
            exit()


    def train (self, estimators=[]):
        self.model.fit(self.X_train_scaled, self.Y_train)
        # Step 4: Retrieve and display coefficients
        self.coefficients = pd.DataFrame({
            'Feature': estimators,
            'Coefficient': self.model.coef_[0]
        })
        self.estimators = estimators

    def predict (self, file=""):
        if not file.empty:
           if not self.singleFile.empty:
              trainFile = self.singleFile
           elif not self.trainFile.empty:
              trainFile = self.trainFile
           missingCols = set(trainFile) - set(file)
           print("missing columns are: \n", missingCols)
           for col in missingCols:
               file[col] = 0
           X = file[self.estimators]
           scaler = StandardScaler()
           self.X_scaled = scaler.fit_transform(X)
        else:
            print("missing file to predict")
            exit()
        self.probabilities = self.model.predict_proba(self.X_scaled)
        return self.model.predict(self.X_scaled)


    def lasso(self, maxIterations=100):
        self.maxIterations = maxIterations
        model = LassoCV(alphas=[1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100], max_iter=self.maxIterations)
        model.fit(self.X_train_scaled, self.Y_train)
        model = Lasso(alpha=model.alpha_, max_iter=self.maxIterations)  # , with Lasso: max_iter=100
        model.fit(self.X_train_scaled, self.Y_train)
        self.iterations = model.n_iter_
        self.predVal = model.predict(self.X_test_scaled)
        return self.predVal

    def ridge(self):
        model = RidgeCV(alphas=[1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100])
        model.fit(self.X_train_scaled, self.Y_train)
        model = Ridge(alpha=model.alpha_, max_iter=100)  # , with Lasso: max_iter=100
        model.fit(self.X_train_scaled, self.Y_train)
        self.predVal = model.predict(self.X_test_scaled)
        return self.predVal

    def logit(self, maxIterations=100):
        X_train_for_statsmodel = sm.add_constant(self.X_train_scaled)
        self.maxIterations = maxIterations
        model = sm.Logit(self.Y_train, X_train_for_statsmodel).fit(maxiter=self.maxIterations)# , family=sm.families.Binomial())
        print("model fitted successfully")
        self.iterations = model.mle_retvals['iterations']
        self.X_test_for_statsmodel = sm.add_constant(self.X_test_scaled)
        self.pvalues = model.pvalues
        self.predVal = model.predict(self.X_test_for_statsmodel)
        return self.predVal

    def RFE(self, model, maxIterations=100, n_features=5):
        self.maxIterations = maxIterations
        #model = LogisticRegression(solver='lbfgs', penalty='l2', max_iter=self.maxIterations)
        # Initialize RFE with Logistic Regression as the model
        rfe = RFE(estimator=model, n_features_to_select=n_features)
        try:
            with warnings.catch_warnings():
                # Ignore Convergence Warnings
                warnings.filterwarnings('ignore', category=ConvergenceWarning)
                rfe.fit(self.X_train_scaled, self.Y_train)

            # Get selected features
            selected_features = rfe.support_
            #print(f"Selected Features: {selected_features}")
            #print(type(selected_features))
            self.selectedFeatures = selected_features
            self.ranking = rfe.ranking_
            # show the coefficients

            # Predict and evaluate
            y_pred = rfe.predict(self.X_test_scaled)
            accuracy = accuracy_score(self.Y_test, y_pred)
            print(f"Accuracy: {accuracy:.4f}")
            self.rfeAcccuracy = accuracy
            #self.selectedEstimators = self.estimators[selected_features]
            self.selectedEstimators = [self.estimators[index] for index in range(len(self.selectedFeatures)) if
                          self.selectedFeatures[index] == True]

        except Exception as e:
            print(f"Error during RFE: {e}")

    def RFE_CV(self, estimator="", step=1, cv=5, scoring='accuracy', predictors=[]):
        # i.e. RFECV(estimator=model, step=1, cv=cv, scoring='accuracy') cv comes from cv = StratifiedKFold(5)  # 5-fold cross-validation: step =1 reduces features one at the time
        # Fit RFECV

        rfecv = RFECV(estimator=estimator, step=step, cv=cv, scoring=scoring)
        X = self.X_train_scaled
        y = self.Y_train

        rfecv.fit(X, y)

        # Number of optimal features
        print("Optimal number of features: %d" % rfecv.n_features_)

        # Accuracy score with optimal number of features
        print("Best cross-validated accuracy:", rfecv.score(X, y))

        # Selected features
        X = pd.DataFrame(X, columns=predictors)
        selected_features = X.columns[rfecv.support_]
        print("Selected features:", selected_features)
        selectedFeatures = selected_features.tolist()
        print(selectedFeatures)

    def SFS (self,
             model="",
             k_features="best",
             forward=True,
             floating=False, # Set True for floating selection (SFFS/SBFS)
             scoring="accuracy", # You can change this to 'roc_auc', 'precision', etc.
             cv=5, # Cross-validation folds
             njobs=-1): # Use all available cores

        sfs = SeqFeatSel (model, k_features, forward, floating, scoring, cv, njobs)

        # Fit SFS on the training data
        print(self.X_train_scaled)
        sfs = sfs.fit(self.X_train_scaled, self.Y_train)

        # The best features are:
        selected_features = sfs.k_feature_names_
        print(f"Selected features: {selected_features}")

        # Transform the data to only use selected features
        X_train_sfs = sfs.transform(self.X_train_scaled)
        X_test_sfs = sfs.transform(self.X_test)

        # Fit logistic regression on the selected features
        model.fit(X_train_sfs, self.Y_train)


        # Predict on the test set
        y_pred = model.predict(X_test_sfs)

        # Evaluate performance
        accuracy = accuracy_score(self.Y_test, y_pred)
        print(f"Accuracy with selected features: {accuracy}")


    def confusionMatrix(self, pred_values, actual_values, title=""):
        pred_values[pred_values<=0.5] = 0
        pred_values[pred_values>0.5] = 1
        conf_matrix = metrics.confusion_matrix(actual_values, np.round(pred_values))
        self.accuracy = metrics.accuracy_score(actual_values, np.round(pred_values))
        print("Accuracy is: ", self.accuracy)
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels = [0, 1]) #, display_labels=[[0, 1],[0,1]]
        cm_display.plot()
        if title != "":
            plt.title(title)
        # Identify indices of false positives and false negatives
        false_positive_indices = np.where((actual_values == 0) & (pred_values == 1))[0]
        false_negative_indices = np.where((actual_values == 1) & (pred_values == 0))[0]

        # Extract the real predicted values (probabilities) for false positives and false negatives
        false_positive_values = pred_values[false_positive_indices]
        false_negative_values = pred_values[false_negative_indices]

        # Print results
        print("False Positive values:", false_positive_values)
        print("False Negative values:", false_negative_values)

        plt.show()

    def wrongPredictions (self):
        columns = self.X_test.columns.tolist()
        data = pd.DataFrame(self.X_test, columns=columns)
        data = data.drop(["Unnamed: 0"], axis=1)
        data["Actual"] = self.Y_test
        data["Predicted"] = np.round(self.predVal)
        self.wrongPredictions = data[data["Actual"] != data["Predicted"]]
        return self.wrongPredictions

    def vif(self, variables, XData=""): #limited use, I am creating another one

        vif = pd.DataFrame()
        '''columns = [col for col in self.X_test.columns if is_numeric_dtype(self.X_test[col]) == True]
        if "Unnamed: 0" in columns:
            columns.remove("Unnamed: 0")'''

        numericData = self.X_train[variables]
        #numericData = numericData.drop(["Unnamed: 0"], axis=1, errors="ignore")

        vif["Attribute"] = variables
        try:
            if len(variables) > 1:
                vif["vif_score"] = [variance_inflation_factor(numericData.values, i) for i in range(len(numericData.columns))]
            else:
                vif["vif_score"] = [variance_inflation_factor(numericData.values, 0)]
        except:
            pass
        #vif["VIF vif_score"] = [variance_inflation_factor(numericData.values, i) for i in range(len(numericData.columns))]
        return vif

    def correlation_matrix(self, x_data="", columns=""):
        data = pd.DataFrame(x_data, columns=columns)
        #print(data)
        correlation_matrix = data.corr()
        return correlation_matrix
    
    def cross_validation(self, X, y, random_state):
        X = pd.DataFrame(X)
        y = pd.Series(y)
        # Initialize KFold

        kf = KFold(n_splits=10, shuffle=True, random_state=random_state)

        # List to store AUC scores for each fold
        auc_scores = []

        # Cross-validation loop
        for train_index, test_index in kf.split(X):
            # Split the data
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            X_train.reset_index(drop=True, inplace=True)
            y_train.reset_index(drop=True, inplace=True)
            X_test.reset_index(drop=True, inplace=True)
            y_test.reset_index(drop=True, inplace=True)

            # Add constant for intercept
            X_train_const = sm.add_constant(X_train)
            X_test_const = sm.add_constant(X_test)

            # Fit the model
            logit_model = sm.Logit(y_train, X_train_const).fit(disp=0)

            # Predict probabilities
            y_pred_prob = logit_model.predict(X_test_const)
            y_pred_prob[y_pred_prob <= 0.8] = 0
            y_pred_prob [y_pred_prob > 0.8] = 1

            # Calculate AUC

            auc = roc_auc_score(y_test, y_pred_prob)
            auc_scores.append(auc)

        # Print the mean AUC score
        print(f"Mean AUC: {np.mean(auc_scores)}")
        print(f"ROC AUC: {auc_scores}")

    def p_value (self):

        try:
            if not warnings.warn(
                ConvergenceWarning("Maximum Likelihood optimization failed to converge. Check mle_retvals ")):
                X_train_for_statsmodel = sm.add_constant(self.X_train_scaled)
                logitModel = sm.Ridge(self.Y_train, X_train_for_statsmodel)
                result = logitModel.fit()
                # Get p-values
                p_values = result.pvalues

                # Filter out features with high p-values (e.g., p > 0.05)
                high_p_value_features = p_values[p_values > 0.05]
                return p_values
        except np.linalg.LinAlgError:
            pass
        except RuntimeWarning or RuntimeError:
            pass



def normalization(item):
    if item > 1:
       item = 1
    return item


class plotDiagram:
    #def __init__(self, data, x_column, y_column, x_label, y_label):
    def __init__(self, x_column, y_column, x_label, y_label):
        plt.scatter(x_column, y_column, alpha=0.2)
        plt.xlabel(f"{x_label}", size=18)
        plt.ylabel(f"{y_label}", size=18)
        #plt.xlim(6, 13)
        #plt.ylim(6, 13)
        plt.show()


def dayAndTimeNorm(x):
     x= 864000*math.floor(x/864000)
     return x



class linearRegression:
    def __init__(self, inputData, features, label):
        if label in features:
            newData = inputData[[col for col in features if col != label] + [label]]
            self.X = newData.drop([label], axis=1).reset_index()
            self.Y = newData[label]
            self.features = features
            self.label = label

    def run (self, X, Y):
        X = sm.add_constant(X)
        linearModel = sm.OLS(Y, X)
        results = linearModel.fit()
        print(f"regression of {self.label} with respect to {self.features} is: ")
        print(results.summary())
        self.lrPvalues = results.pvalues
        vif = pd.DataFrame()
        vif["Feature"] = X.columns
        vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        self.lrVif = vif


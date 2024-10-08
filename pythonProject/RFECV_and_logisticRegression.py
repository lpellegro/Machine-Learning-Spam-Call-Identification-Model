from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from models import *
import joblib
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression
import joblib

# environment variables
load_dotenv()

pd.set_option('display.max_columns', None)
workingDir = os.getenv("ANALYSIS_DIRECTORY")
cdrDir = os.getenv("CDR_DIR")

pd.set_option('display.max_columns', None)

# reading the CDR file and treat categorical and binary variables
readFile = filePreparation(filename='all_calls.csv', workingDir=workingDir, cdrDir=cdrDir)

# dropping unused columns
data = readFile.numericData.drop(["Day and time (sec)", "Source IP (integer)"], axis = 1)

# keeping only estimators with numerical or boolean values
estimators = readFile.numericEstimators
print("estimators are ", estimators)
estimators.remove("Day and time (sec)")
estimators.remove("Source IP (integer)")

# creating a logistic regression model and split the data into train and test against all numeric estimators
model = regressionModel(LogisticRegression(solver="lbfgs", max_iter=100, penalty="l2"))
model.trainTestGen(file=data, estimators=estimators)

# Performing RFECV to find the best combination of predictors
rfeModel = LogisticRegression(penalty='l2', solver='lbfgs', max_iter=1000)

# Cross-validation strategy
cv = StratifiedKFold(5)  # 5-fold cross-validation
rfeCv = model.RFE_CV(estimator=rfeModel, cv=cv, predictors=estimators)

# selecting the most stable estimators
estimators = ["Traversal", "Source IP Location"]

# splitting the train and test data with respect to only new estimators
model.trainTestGen(file=data, estimators=estimators)
model.train(estimators=estimators)
print(model.coefficients)

# save the trained model
joblib.dump(model, f'{workingDir}/2_predictor_joblib_model.pkl')

# predict the data
fileToPredict = filePreparation(filename="all_calls_2024-10-04.csv", workingDir=workingDir, cdrDir=cdrDir)
predictionFile = fileToPredict.numericData.drop(["Day and time (sec)", "Source IP (integer)"], axis = 1)
result = model.predict(predictionFile)
Y = fileToPredict.actualValues("Pass")
probabilities = model.probabilities
model.confusionMatrix(result, Y, "all predictors")
results = ['y' if item == 1 else "n" for item in result]

# appending predicted values and scores to the data
originalFile = fileToPredict.data
originalFile["Predict"] = results
originalFile["Pass"] = originalFile["Pass"].map({1: "y", 0: "n"})
originalFile["y score"] = probabilities[:, 1]
originalFile["n score "] = probabilities[:, 0]
differences = originalFile[originalFile["Pass"] != originalFile["Predict"]]

# showing the errors
differences.to_csv(f"{workingDir}/differences.csv")


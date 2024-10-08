import joblib
from dotenv import load_dotenv
import os
from models import *

load_dotenv()

pd.set_option('display.max_columns', None)
workingDir = os.getenv("ANALYSIS_DIRECTORY")
cdrDir = os.getenv("CDR_DIR")


model = joblib.load(f"{workingDir}/2_predictor_joblib_model.pkl")
print(model.estimators)
fileToPredict = filePreparation(filename="all_calls_2024-10-04.csv", workingDir=workingDir, cdrDir=cdrDir)
predictionFile = fileToPredict.numericData.drop(["Day and time (sec)", "Source IP (integer)"], axis = 1)

result = model.predict(predictionFile)
Y = fileToPredict.actualValues("Pass")
print(model.probabilities)
model.confusionMatrix(result, Y, "all predictors")




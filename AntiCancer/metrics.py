import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import math

def evaluate(prediction, target):
    mse = mean_squared_error(target, prediction)
    rmse = math.sqrt(mse)
    R2 = r2_score(target, prediction)
    pearson = np.corrcoef(target, prediction)[0, 1]
    fit = pearson + R2 - rmse 
    return {
        "mse" : mse,
        "rmse" : rmse,
        "r2" : R2,
        "pear" : pearson,
        "fit" : fit
    }
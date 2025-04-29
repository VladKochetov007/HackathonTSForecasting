import pandas as pd
import numpy as np
from sklearn.metrics import r2_score

preds = pd.read_csv('lfs_data/final_result.csv')
actual = pd.read_csv('Timeseries_six_test.csv')

r_2 = []

for i in range(1, 7):
    r_2.append(r2_score(preds[f"{i-1}"], actual[f"Series{i}"]))

print(f"R2 score for each series: {r_2}")
print(f"Average R2 score: {np.mean(r_2):.4f}")
print(f"Average R2 root: {np.sqrt(np.mean(r_2)):.4f}")
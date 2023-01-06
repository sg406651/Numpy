import numpy as np

data = np.load("ex3_data.npy")
print("Total number of NaN values in the data:", np.sum(np.isnan(data)))
dataModified = data[~np.isnan(data).any(axis=1)]
print("Rows with NaN value: \n", data[np.isnan(data).any(axis=1)])
print("Number of rows with NaN values:", np.size(data[np.isnan(data).any(axis=1)], 0))
print("Number of NaN values in each column:", np.sum(np.isnan(data), axis=0))

import pandas as pd
import numpy as np

training_data_filepath = "data/train.csv"
chunk_size = 1000

for chunk in pd.read_csv(training_data_filepath, chunksize=chunk_size):
	
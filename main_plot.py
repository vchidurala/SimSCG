# -*- coding: utf-8 -*-
from utils import *

# Old data
data_file = './data/simu_10000_0.1_140_180.npy'
data_path = './data/simu_10000_0.1_140_180.npy'

# New Data
# Uncomment below two lines to run it on new dataset & commenting above old data files
# data_file = './data/day_data_test.npy'
# data_path = './data/day_data_test.npy'

sp_prediction, sp_MAE, dp_prediction, dp_MAE = prediction(data_file)
prediction_file = './data/prediction.npy'  # To save the predicted data
# np.save(prediction_file, [sp_prediction, dp_prediction])
print('===============')
print('Systolic MAE:', sp_MAE)
print('===============')
print('Diastolic MAE:', dp_MAE)
print('===============')

data_set = np.load(data_path)
sp_label = data_set[:, -2]
dp_label = data_set[:, -1]

plot_2vectors(sp_label, sp_prediction, 'sp')
plot_2vectors(dp_label, dp_prediction, 'dp')

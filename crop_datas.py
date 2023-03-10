import numpy as np
import os

dir = os.getcwd()
if "\\" in dir:
    prefix = "\\"
elif "/" in dir:
    prefix = "/"
else:
    prefix = ""

dirTrain = prefix.join((dir, "data", "elect", "train_data_elect.npy"))
dirTrainLab = prefix.join((dir, "data", "elect", "train_label_elect.npy"))
dirTrainV = prefix.join((dir, "data", "elect", "train_v_elect.npy"))

dirTest = prefix.join((dir, "data", "elect", "test_data_elect.npy"))
dirTestLab = prefix.join((dir, "data", "elect", "test_label_elect.npy"))
dirTestV = prefix.join((dir, "data", "elect", "test_v_elect.npy"))

dirValid = prefix.join((dir, "data", "elect", "valid_data_elect.npy"))
dirValidLab = prefix.join((dir, "data", "elect", "valid_label_elect.npy"))
dirValidV = prefix.join((dir, "data", "elect", "valid_v_elect.npy"))

outTrain = np.load(dirTrain)
outTrainLab = np.load(dirTrainLab)
outTrainV = np.load(dirTrainV)

outTest = np.load(dirTest)
outTestLab = np.load(dirTestLab)
outTestV = np.load(dirTestV)

outValid = np.load(dirValid)
outValidLab = np.load(dirValidLab)
outValidV = np.load(dirValidV)

print(f'TRAIN : {outTrain.shape}')
print(f'TEST : {outTest.shape}')
print(f'VALID : {outValid.shape}')
print(f'TRAIN-LABELS : {outTrainLab.shape}')
print(f'TRAIN-V : {outTrainV.shape}')
print(outTrainV)

# Crop training data
np.save('train_data_elect.npy', outTrain[:20, :, :])
np.save('train_label_elect.npy', outTrainLab[:20, :])
np.save('train_v_elect.npy', outTrainV[:20, :])
# Crop testing data
np.save('test_data_elect1.npy', outTest[:10, :, :])
np.save('test_label_elect1.npy', outTestLab[:10, :])
np.save('test_v_elect1.npy', outTestV[:10, :])
# Crop validation data
np.save('valid_data_elect1.npy', outValid[:10, :, :])
np.save('valid_label_elect1.npy', outValidLab[:10, :])
np.save('valid_v_elect1.npy', outValidV[:10, :])

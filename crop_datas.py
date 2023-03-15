import numpy as np
import os
import shutil

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

try:
    shutil.rmtree("Crop_datas")
except FileNotFoundError:
    pass
# create missing directories
try:
    os.mkdir("Crop_datas")
except FileExistsError:
    pass

# Crop training data
np.save('Crop_datas'+prefix+'train_data_elect.npy', outTrain[:3000, :, :])
np.save('Crop_datas'+prefix+'train_label_elect1.npy', outTrainLab[:3000, :])
np.save('Crop_datas'+prefix+'train_v_elect.npy', outTrainV[:3000, :])
# Crop testing data
np.save('Crop_datas'+prefix+'test_data_elect.npy', outTest[:800, :, :])
np.save('Crop_datas'+prefix+'test_label_elect.npy', outTestLab[:800, :])
np.save('Crop_datas'+prefix+'test_v_elect.npy', outTestV[:800, :])
# Crop validation data
np.save('Crop_datas'+prefix+'valid_data_elect.npy', outValid[:800, :, :])
np.save('Crop_datas'+prefix+'valid_label_elect.npy', outValidLab[:800, :])
np.save('Crop_datas'+prefix+'valid_v_elect.npy', outValidV[:800, :])

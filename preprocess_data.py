from __future__ import absolute_import
import matplotlib.pyplot as plt

import os
import argparse
from datetime import datetime, timedelta
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import math
import numpy as np
import random
from tqdm import trange

from IPython import embed
from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile

from math import sqrt
from pandas import read_csv, DataFrame
from scipy import stats

import utils

import matplotlib
matplotlib.use('Agg')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='elect', help='Name of the dataset')
parser.add_argument('--model-name', default='transformermodify_model',
                    help='Directory containing params.json')
parser.add_argument('--data-folder', default='data',
                    help='Parent dir of the dataset')


def prep_data(data, params: utils.Params, data_frame, covariates, data_start, train=False, valid=False, test=False):
    time_len = data.shape[0]
    total_time = data_frame.shape[0]
    num_series = data_frame.shape[1]  # 370 for ELECT

    input_size = params.train_window-params.stride_size
    windows_per_series = np.full(
        (num_series), (time_len-input_size) // params.stride_size)

    if train:
        windows_per_series -= (data_start +
                               params.stride_size-1) // params.stride_size

    total_windows = np.sum(windows_per_series)
    x_input = np.zeros((total_windows, params.train_window, 1 +
                       params.num_covariates + 1), dtype='float32')
    label = np.zeros((total_windows, params.train_window), dtype='float32')
    v_input = np.zeros((total_windows, 2), dtype='float32')
    count = 0

    if not train:
        covariates = covariates[-time_len:]
    for series in trange(num_series):
        cov_age = stats.zscore(np.arange(total_time-data_start[series]))
        #cov_age = np.arange(total_time-data_start[series])
        if train:
            covariates[data_start[series]:time_len,
                       0] = cov_age[:time_len-data_start[series]]
        else:
            covariates[:, 0] = cov_age[-time_len:]

        for i in range(windows_per_series[series]):
            if train:
                window_start = params.stride_size*i+data_start[series]
            else:
                window_start = params.stride_size*i
            window_end = window_start+params.train_window
            x_input[count, 1:, 0] = data[window_start:window_end-1, series]
            x_input[count, :, 1:1 +
                    params.num_covariates] = covariates[window_start:window_end, :]
            x_input[count, :, -1] = series
            label[count, :] = data[window_start:window_end, series]
            nonzero_sum = (x_input[count, 1:input_size, 0] != 0).sum()
            if nonzero_sum == 0:
                v_input[count, 0] = 0
            else:
                v_input[count, 0] = np.true_divide(
                    x_input[count, 1:input_size, 0].sum(), nonzero_sum)+1
                x_input[count, 1:, 0] = (
                    x_input[count, 1:, 0] - v_input[count, 1]) / v_input[count, 0]
                if train:
                    label[count, :] = (label[count, :] -
                                       v_input[count, 1])/v_input[count, 0]
            count += 1

    if train:
        prefix = os.path.join(save_path, 'train_')
    elif valid:
        prefix = os.path.join(save_path, 'valid_')
    else:
        prefix = os.path.join(save_path, 'test_')

    np.save(prefix+'data_'+args.dataset, x_input)
    np.save(prefix+'v_'+args.dataset, v_input)
    np.save(prefix+'label_'+args.dataset, label)


def gen_covariates(times, params: utils.Params):
    dims = params.num_covariates
    covariates = np.zeros((times.shape[0], dims))
    for i, input_time in enumerate(times):
        covariates[i, 1] = input_time.weekday()  # 6
        covariates[i, 2] = input_time.hour  # 24

    for i in range(1, dims):
        covariates[:, i] = stats.zscore(covariates[:, i])
    return covariates


if __name__ == '__main__':
    global save_path

    args = parser.parse_args()
    model_dir = os.path.join('experiments', args.model_name)
    json_path = os.path.join(model_dir, 'params.json')
    data_dir = os.path.join(args.data_folder, args.dataset)
    assert os.path.isfile(
        json_path), f'No json configuration file found at {json_path}'
    params = utils.Params(json_path)
    # Replaced by parameters in param.json (look the line above)
    # window_size = 192  # pre 7 conditional,next 1 day prediction
    # stride_size = 24

    name = 'LD2011_2014.txt'

    train_start = '2014-01-01 00:00:00'
    train_end = '2014-12-18 00:00:00'
    valid_start = '2014-12-11 00:00:00'
    valid_end = '2014-12-24 23:00:00'
    test_start = '2014-12-18 00:00:00'  # need additional 7 days as given info
    test_end = '2014-12-31 23:00:00'

    save_path = os.path.join(args.data_folder, args.dataset)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    initial_path = os.getcwd()

    # The csv_path (contening ELECT datas for exemple) can be :
    csv_path = os.path.join(initial_path, args.data_folder,
                            args.dataset, name)

    if not os.path.exists(csv_path):
        zipurl = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00321/LD2011_2014.txt.zip'
        with urlopen(zipurl) as zipresp:
            with ZipFile(BytesIO(zipresp.read())) as zfile:
                zfile.extractall(save_path)

    data_frame = pd.read_csv(
        csv_path, sep=";", index_col=0, parse_dates=True, decimal=',')
    data_frame = data_frame.resample('1H', label='left', closed='right').sum()[
        train_start:test_end]
    data_frame.fillna(0, inplace=True)

    covariates = gen_covariates(
        data_frame[train_start:test_end].index, params)
    train_data = data_frame[train_start:train_end].values
    valid_data = data_frame[valid_start: valid_end].values
    test_data = data_frame[test_start:test_end].values
    # find first nonzero value in each time series
    data_start = (train_data != 0).argmax(axis=0)

    prep_data(train_data, params, data_frame,
              covariates, data_start, train=True)
    prep_data(valid_data, params, data_frame,
              covariates, data_start, valid=True)
    prep_data(test_data, params, data_frame, covariates, data_start, test=True)

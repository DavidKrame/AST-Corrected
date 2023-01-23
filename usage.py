import numpy as np
import logging
import tqdm
import os
import argparse
import copy
import matplotlib.pyplot as plt
from torch.utils.data.sampler import RandomSampler

import torch

import utils
from dataloader import *
from evaluate import *
import gan_transformer as transformer

logger = logging.getLogger('Transformer.Usage')
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='elect', help='Name of the dataset')
parser.add_argument('--data-folder', default='data',
                    help='Parent dir of the dataset')
parser.add_argument('--model-name', default='test',
                    help='Directory containing params.json and best.pth.tar')

parser.add_argument('--restore-file', default='best',
                    help='Optional, name of the file in --model_dir containing weights to reload before \
                    training')  # 'best' or 'epoch_#'
parser.add_argument('--usage-fig', default='figures',
                    help='Dir that contain a model output figures. By default : figures directory')  # 'best' or 'epoch_#'


def usage(model, test_loader, params):
    '''Usage of the model.
    Args:
        model: (torch.nn.Module) the Deep AR model
        test_loader: load test data and labels
        params: (Params) hyperparameters
    '''
    model.eval()
    with torch.no_grad():
        """NOTE : You can decrease a predict_steps in params.json (/experiments/test)"""
        sum_mu = torch.zeros([740, params.predict_steps]).to(params.device)

        true = torch.zeros([740, params.predict_steps]).to(params.device)

        for i, (test_batch, id_batch, v, labels) in enumerate(tqdm(test_loader)):
            test_batch = test_batch.to(torch.float32).to(params.device)
            id_batch = id_batch.unsqueeze(-1).to(params.device)
            v_batch = v.to(torch.float32).to(params.device)
            labels = labels.to(torch.float32).to(params.device)

            sample_mu, sample_q90 = transformer.test(
                model, params, test_batch, v_batch, id_batch)

            if(i == 0):
                sum_mu = sample_mu
                sum_q90 = sample_q90
                true = labels[:, -params.predict_steps:]
            else:
                sum_mu = torch.cat([sum_mu, sample_mu], 0)
                sum_q90 = torch.cat([sum_q90, sample_q90], 0)
                true = torch.cat([true, labels[:, -params.predict_steps:]], 0)

    return sum_mu, sum_q90, true


if __name__ == "__main__":
    # Load the parameters
    args = parser.parse_args()
    model_dir = os.path.join('experiments', args.model_name)
    json_path = os.path.join(model_dir, 'params.json')
    data_dir = os.path.join(args.data_folder, args.dataset)
    assert os.path.isfile(
        json_path), 'No json configuration file found at {}'.format(json_path)
    params = utils.Params(json_path)

    cuda_exist = torch.cuda.is_available()  # use GPU is available

    # Set random seeds for reproducible experiments if necessary

    params.device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu")
    # torch.cuda.manual_seed(240)
    logger.info('Using Cuda...')

    c = copy.deepcopy
    attn = transformer.MultiHeadedAttention(params)
    ff = transformer.PositionwiseFeedForward(
        params.d_model, d_ff=params.d_ff, dropout=params.dropout)
    position = transformer.PositionalEncoding(
        params.d_model, dropout=params.dropout)
    ge = transformer.Generator(params)
    emb = transformer.Embedding(params, c(position))

    model = transformer.EncoderDecoder(params=params, emb=emb, encoder=transformer.Encoder(params, transformer.EncoderLayer(params, c(attn), c(
        ff), dropout=params.dropout)), decoder=transformer.Decoder(params, transformer.DecoderLayer(params, c(attn), c(attn), c(ff), dropout=params.dropout)), generator=ge)

    model.to(params.device)

    # Create the input data pipeline
    logger.info('Loading the test_set...')

    test_set = TestDataset(data_dir, args.dataset, params.num_class)
    test_loader = DataLoader(test_set, batch_size=params.predict_batch,
                             sampler=RandomSampler(test_set), num_workers=4)
    logger.info('- done.')

    # print('model: ', model)

    logger.info('Starting usage')

    # Reload weights from the saved file
    utils.load_checkpoint(os.path.join(
        model_dir, args.restore_file + '.pth.tar'), model)

    sum_mu, sum_q90, true = usage(model, test_loader, params)

    # create missing directories
    try:
        os.mkdir(args.usage_fig)
    except FileExistsError:
        pass

    """NOTE : You can decrease a predict_steps in params.json (/experiments/test)"""
    for k in range(true.shape[0]):
        save_fig_location = os.path.join(args.usage_fig, f"test{k+1}.png")
        # Data for plotting
        labels = true[k, :]
        predictions = sum_mu[k, :]
        t = np.arange(len(predictions))

        plt.figure(k)
        plt.plot(t, predictions, label="predictions")
        plt.plot(t, labels, label="labels")

        plt.xlabel = 'time'
        plt.ylabel = 'values'
        plt.ylim(0, 150)
        plt.title = f'Predictions {k}'
        plt.legend()
        plt.grid()

        plt.savefig(save_fig_location)
        plt.show()

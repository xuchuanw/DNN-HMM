import numpy as np
import pandas as pd
import logging
from pandas import DataFrame, Series
import dnnhmm
import argparse
import pickle
import sys

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_feats', type=str,
                        default='./feats/english_train_lib.pkl', help='training data feats path')
    parser.add_argument('--test_feats', type=str,
                        default='./feats/english_test_lib.pkl', help='test data feats path')
    parser.add_argument('--niter', type=int, default=10)
    parser.add_argument('--nstate', type=int, default=5)
    parser.add_argument('--nepoch', type=int, default=8)
    parser.add_argument('--lr', type=int, default=0.01)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--hmm_mddel_save', type=str, default='hmm.pickle')

    parser.add_argument('--mode', type=str, default='mlp',
                        choices=['hmm', 'mlp'],
                        help='Type of models')

    args = parser.parse_args()
    lr = 0.01

    # logging info
    log_format = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s:%(message)s"
    logging.basicConfig(level=logging.INFO, format=log_format)
    np.random.seed(777)

    utt_train = pd.read_pickle(args.train_feats)
    utt_test = pd.read_pickle(args.test_feats)
    digits = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    uniq_state_dct = {}
    i = 0
    for digit in digits:
        for state in range(args.nstate):
            uniq_state_dct[(digit, state)] = i
            i += 1

    train_data_dct = {}
    for d in digits:
        train_data_dct[d] = []
    for index, utt_data in utt_train.iterrows():
        train_data_dct[utt_data['label']].append(utt_data['feats'])

    # Single Gaussian

    sg_model = dnnhmm.sg_train(digits, train_data_dct)

    if args.mode == 'hmm':
        try:
            model = pickle.load(open(args.hmm_mddel_save, 'rb'))
        except:
            model = dnnhmm.hmm_train(
                digits, train_data_dct, sg_model, args.nstate, args.niter)
            pickle.dump(model, open(args.hmm_mddel_save, 'wb'))
    elif args.mode == 'mlp':
        try:
            hmm_model = pickle.load(open(args.hmm_mddel_save, 'rb'))

        except:
            hmm_model = dnnhmm.hmm_train(
                digits, train_data_dct, sg_model, args.nstate, args.niter)
            pickle.dump(hmm_model, open(args.hmm_mddel_save, 'wb'))

        model = dnnhmm.mlp_train(digits, train_data_dct, hmm_model, uniq_state_dct, lr=args.lr,
                                 nunits=(64, 64))

    # test
    total_count = 0
    correct = 0
    for index, utt_data in utt_test.iterrows():

        lls = []
        for digit in digits:
            ll = model[digit].loglike(utt_data['feats'], digit)

            lls.append(ll)

        predict = digits[np.argmax(np.array(lls))]
        log_like = np.max(np.array(lls))
        print(predict, log_like)

        logging.info("predict %s for utt %s (log like = %f)",
                     predict, utt_data['label'], log_like)
        if str(predict) == utt_data['label']:
            correct += 1
        total_count += 1

    logging.info("accuracy: %f", float(correct)/total_count * 100)

# -*- coding: utf-8 -*-
from io import open
import unicodedata
import string
import re
from args import args
import random

import matplotlib.pyplot as plt
plt.switch_backend('TkAgg')
import matplotlib.ticker as ticker
import numpy as np

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from language import prepareData
from model import EncoderRNN, AttnDecoderRNN
from evaluation import evaluateRandomly, evaluate
from plot import evaluateAndShowAttention
from train import trainIters
from translator import Translator


def main():
    input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
    print(random.choice(pairs))

    device = torch.device(args.device)
    print('device : {}'.format(device))

    encoder = EncoderRNN(input_lang.n_words, args.hidden_size).to(device)
    decoder = AttnDecoderRNN(args.hidden_size, output_lang.n_words, dropout_p=0.1).to(device)

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=args.lr)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=args.lr)

    model = Translator(input_lang, output_lang, encoder, decoder, encoder_optimizer, decoder_optimizer)

    trainIters(model, pairs, n_iters=10000, print_every=100, plot_every=100)

    evaluateRandomly(model, pairs)

    output_words, attentions = evaluate(model, "je suis trop froid .")
    plt.matshow(attentions.numpy())

    # evaluateAndShowAttention("elle a cinq ans de moins que moi .")
    #
    # evaluateAndShowAttention("elle est trop petit .")
    #
    # evaluateAndShowAttention("je ne crains pas de mourir .")
    #
    # evaluateAndShowAttention("c est un jeune directeur plein de talent .")


if __name__ == '__main__':
    main()

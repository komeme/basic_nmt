import torch
from args import args
import random
import time
from language import tensorsFromPair
from utils import timeSince, asMinutes
from plot import showPlot
from language import prepareData

device = torch.device(args.device)


def trainIters(translator, pairs, n_iters, print_every=1000, plot_every=100):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    training_pairs = [tensorsFromPair(translator.input_lang, translator.output_lang, random.choice(pairs)) for _ in range(n_iters)]

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = translator.train(input_tensor, target_tensor)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        # if iter % plot_every == 0:
        #     plot_loss_avg = plot_loss_total / plot_every
        #     plot_losses.append(plot_loss_avg)
        #     plot_loss_total = 0

    showPlot(plot_losses)


import torch
import random
from args import args
from language import tensorFromSentence
from setting import SOS_token, EOS_token

device = torch.device(args.device)


def evaluate(translator, sentence, max_length=args.max_length):
    input_tensor = tensorFromSentence(translator.input_lang, sentence)
    return translator.translate(input_tensor)

def evaluateRandomly(translator, pairs, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(translator, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')

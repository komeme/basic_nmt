from args import args
import torch
from torch import nn
from setting import SOS_token, EOS_token, teacher_forcing_ratio
import random
import time
from torch import optim

MAX_LENGTH = args.max_length
device = torch.device(args.device)
teacher_forcing_ratio = args.teacher_forcing_ratio


class Translator:
    def __init__(self, input_lang, output_lang, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion=nn.NLLLoss()):
        self.input_lang = input_lang
        self.output_lang = output_lang
        self.encoder = encoder
        self.decoder = decoder
        self.encoder_optimizer = encoder_optimizer
        self.decoder_optimizer = decoder_optimizer
        self.criterion = criterion

    def train(self, input_tensor, target_tensor):
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        encoder_outputs, encoder_hidden = self._encode(input_tensor)

        target_length = target_tensor.size(0)

        loss = 0

        decoder_input = torch.tensor([[SOS_token]], dtype=torch.long, device=device)

        decoder_hidden = encoder_hidden

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                loss += self.criterion(decoder_output, target_tensor[di])
                decoder_input = target_tensor[di]  # Teacher forcing

        else:
            # Without teacher forcing: use its own predictions as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()  # detach from history as input

                loss += self.criterion(decoder_output, target_tensor[di])
                if decoder_input.item() == EOS_token:
                    break

        loss.backward()

        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        return loss.item() / target_length

    def translate(self, input_tensor):
        with torch.no_grad():
            encoder_outputs, encoder_hidden = self._encode(input_tensor)

            decoder_input = torch.tensor([[SOS_token]], dtype=torch.long, device=device)  # SOS

            decoder_hidden = encoder_hidden

            decoded_words = []
            decoder_attentions = torch.zeros(MAX_LENGTH, MAX_LENGTH)

            for di in range(MAX_LENGTH):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                decoder_attentions[di] = decoder_attention.data
                topv, topi = decoder_output.data.topk(1)
                if topi.item() == EOS_token:
                    decoded_words.append('<EOS>')
                    break
                else:
                    decoded_words.append(self.output_lang.index2word[topi.item()])

                decoder_input = topi.squeeze().detach()

            return decoded_words, decoder_attentions[:di + 1]

    def _encode(self, input_tensor):
        input_length = input_tensor.size()[0]
        encoder_hidden = self.encoder.initHidden()

        encoder_outputs = torch.zeros(MAX_LENGTH, self.encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        return encoder_outputs, encoder_hidden


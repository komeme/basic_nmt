import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--max_length', default=10, help='max lenght of input sentence')
parser.add_argument('--device', default='cpu', choices=['cpu', 'gpu'], help='device to run')
parser.add_argument('--lang1', default='eng', help='input language')
parser.add_argument('--lang2', default='fra', help='output language')
parser.add_argument('--hidden_size', default=256, help='hidden size of RNN')
parser.add_argument('--teacher_forcing_ratio', default=1.0, help='teacher forcing ratio')
parser.add_argument('--lr', default=0.01, help='learning rate')

args = parser.parse_args()

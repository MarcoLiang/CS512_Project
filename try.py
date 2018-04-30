import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

word_embed = np.load('./embedding_file/w2v/w2vembedd_128.npy')

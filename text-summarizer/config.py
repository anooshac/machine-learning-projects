from os import path, makedirs

path_src = path.dirname(path.abspath(__file__))
path_data = path.join(path_src, 'data/article-data')
path_logs = path.join(path_src, 'logs')
path_outputs = path.join(path_src, 'outputs')
path_glove = path.join(path_src, 'data/glove-values')

#verify that paths exist, otherwise create directories
for p in [path_data, path_logs, path_outputs]:
    if not path.exists(p):
        makedirs(p)


#training parameters
#random_seed = 42
#batch_size = 4
epochs = 10
max_length_description = 100 #max length of description
max_length_heading = 15 #max length of heading
rnn_size = 512 #size of RNN layers
rnn_layers = 3 #number of RNN layers
num_samples = 640 #number of samples per epoch
num_flips = 10 #number of flips
rnn_temperature = 0.8
learning_rate = 0.0001
print_freq = 10
batch_norm = False

negative_sample = 70
loss_eval_freq = 100
similarity_eval_freq = 100
nearest_neighbours = 8


window_size = 2

# Special vocabulary symbols - we always put them at the start.
_PAD = b"_PAD"
_GO = b"_GO"
_EOS = b"_EOS"
_UNK = b"_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

#new training parameters
random_seed = 42
batch_size = 32
embedding_dim = 512
epochs = 1000
learning_rate = 0.01
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

#new training parameters
random_seed = 42
batch_size = 32
embedding_dim = 512
epochs = 1000
learning_rate = 0.01
print_freq = 2

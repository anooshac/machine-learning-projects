import json
import config
from os import path
from nltk.tokenize import word_tokenize as tokenize
import nltk
import itertools
import numpy as np
import cPickle as pickle

WHITELIST = '0123456789abcdefghijklmnopqrstuvwxyz '
VOCAB_SIZE = 1200
UNK = 'unk'

limit = {
    'max_descriptions' : 400,
    'min_descriptions' : 0,
    'max_headings' : 20,
    'min_headings' : 0,
}

def load_raw_data(filename):
    #################################
    #   Loads raw data from file    #
    #################################

    with open(filename, 'r') as fp:
        raw_data = json.load(fp)

    print('Loaded {:,} articles from {}'.format(len(raw_data), filename))
    return raw_data

def tokenize_sentence(sentence):
    ######################################
    #   Splits article into sentences    #
    ######################################

    return ' '.join(list(tokenize(sentence)))

def article_is_complete(article):
    ###########################################################
    #   Checks if article has both heading and description    #
    ###########################################################

    if ('abstract' not in article) or ('article' not in article):
        return False
    if (article['abstract'] is None) or (article['article'] is None):
        return False

    return True

def tokenize_articles(raw_data):
    #########################################################################
    #   Tokenizes raw data and creates list of headings and descriptions    #
    #########################################################################

    headings, descriptions = [], []
    num_articles = len(raw_data)

    for i, a in enumerate(raw_data):
        if article_is_complete(a):
            headings.append(tokenize_sentence(a['abstract']))
            descriptions.append(tokenize_sentence(a['article']))
        if i % config.print_freq == 0:
            print('Tokenized {:,} / {:,} articles'.format(i, num_articles))

    return (headings, descriptions)

def filter(line, whitelist):
    ##############################################################
    #   Filters out all characters which are not in whitelist    #
    ##############################################################

    return ''.join([ch for ch in line if ch in whitelist])

def filter_length(headings, descriptions):
    ######################################################################
    #   Filters based on heading and description length defined above    #
    ######################################################################

    if len(headings) != len(descriptions):
        raise Exception('Number of headings does not match number of descriptions!')

    filtered_headings, filtered_descriptions = [], []

    for i in range(0, len(headings)):
        heading_length = len(headings[i].split(' '))
        description_length = len(descriptions[i].split(' '))

        if description_length >= limit['min_descriptions'] and description_length <= limit['max_descriptions']:
            if heading_length >= limit['min_headings'] and heading_length <= limit['max_headings']:
                filtered_headings.append(headings[i])
                filtered_descriptions.append(descriptions[i])

    print ('Length of filtered headings: {:,}'.format(len(filtered_headings)))
    print ('Length of filtered descriptions: {:,}'.format(len(filtered_descriptions)))

    return (filtered_headings, filtered_descriptions)

def index_data(tokenized_sentences, vocab_size):
    #####################################################
    #   Forms vocab, and idx2word and word2idx dicts    #
    #####################################################

    freq_dist = nltk.FreqDist(itertools.chain(*tokenized_sentences))
    vocab = freq_dist.most_common(vocab_size)
    print ('Vocab length: {:,}'.format(len(vocab)))

    idx2word = ['_'] + [UNK] + [x[0] for x in vocab]
    word2idx = dict([(w, i) for i, w in enumerate(idx2word)])

    return (idx2word, word2idx, freq_dist)

def pad_seq(seq, lookup, max_length):
    #########################################
    #   Pads sequence with zero values      #
    #########################################

    indices = []

    for word in seq:
        if word in lookup:
            indices.append(lookup[word])
        else:
            indices.append(lookup[UNK])

    return indices + [0]*(max_length - len(seq))

def zero_pad(tokenized_headings, tokenized_descriptions, word2idx):
    #############################################
    #   Stores indices in numpy arrays and      #
    #   creates zero padding where required     #
    #############################################
    data_length = len(tokenized_descriptions)

    idx_descriptions = np.zeros([data_length, limit['max_descriptions']], dtype=np.int32)
    idx_headings = np.zeros([data_length, limit['max_headings']], dtype=np.int32)

    for i in range(data_length):
        description_indices = pad_seq(tokenized_descriptions[i], word2idx, limit['max_descriptions'])
        heading_indices = pad_seq(tokenized_headings[i], word2idx, limit['max_headings'])

        idx_descriptions[i] = np.array(description_indices)
        idx_headings[i] = np.array(heading_indices)

    return (idx_headings, idx_descriptions)

def process_data():

    #load data from file
    filename = path.join(config.path_data, 'raw_data.json')
    raw_data = load_raw_data(filename)

    #tokenize articles and separate into headings and descriptions
    headings, descriptions = tokenize_articles(raw_data)

    #keep only whitelisted characters and articles satisfying the length limits
    headings = [filter(heading, WHITELIST) for heading in headings]
    descriptions = [filter(sentence, WHITELIST) for sentence in descriptions]
    headings, descriptions = filter_length(headings, descriptions)

    #convert list of sentences into list of list of words
    word_tokenized_headings = [word_list.split(' ') for word_list in headings]
    word_tokenized_descriptions = [word_list.split(' ') for word_list in descriptions]

    #indexing
    idx2word, word2idx, freq_dist = index_data(word_tokenized_headings + word_tokenized_descriptions, VOCAB_SIZE)

    #save as numpy array and do zero padding
    idx_headings, idx_descriptions = zero_pad(word_tokenized_headings, word_tokenized_descriptions, word2idx)

    #check percentage of unks
    unk_percentage = calculate_unk_percentage(idx_headings, idx_descriptions, word2idx)
    print (calculate_unk_percentage(idx_headings, idx_descriptions, word2idx))

    article_data = {
        'word2idx' : word2idx,
        'idx2word': idx2word,
        'limit': limit,
        'freq_dist': freq_dist,
    }

    pickle_data(article_data)

    return (idx_headings, idx_descriptions)

def pickle_data(article_data):
    ###########################################
    #   Saves obj to disk as a pickle file    #
    ###########################################

    with open(path.join(config.path_data, 'article_data.pkl'), 'wb') as fp:
        pickle.dump(article_data, fp, 2)

def unpickle_articles():
    #################################################
    #   Loads pickle file from disk to give obj     #
    #################################################

    with open(path.join(config.path_data, 'article_data.pkl'), 'rb') as fp:
        article_data = pickle.load(fp)

    return article_data

def calculate_unk_percentage(idx_headings, idx_descriptions, word2idx):
    num_unk = (idx_headings == word2idx[UNK]).sum() + (idx_descriptions == word2idx[UNK]).sum()
    num_words = (idx_headings > word2idx[UNK]).sum() + (idx_descriptions > word2idx[UNK]).sum()

    return (num_unk / num_words) * 100

def main():
    process_data()

if __name__ == '__main__':
    main()

import sys
import pandas as pd
import numpy as np
import pickle

sys.path.insert(0, '/')
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from helper.dataset_reader import read_tsv


# ROOT_DATA = '../dataset/ijelid_peerj_cs'

def load_dataset(data_path):
    all_data = read_tsv(f'{data_path}/ijelid_clean_11K.tsv')
    train_data = read_tsv(f'{data_path}/train.tsv')
    val_data = read_tsv(f'{data_path}/val.tsv')
    test_data = read_tsv(f'{data_path}/test.tsv')

    return all_data, train_data, val_data, test_data


def build_dictionary(all_data):
    # build words dictionary and tags dictionary
    tokens = list(set(all_data[1]))
    tokens.append("ENDPAD")
    tags = np.unique(all_data[2])

    return tokens, tags


def input_converter(model_name, input_data, words, tags, w_embd=True, wc_embd=True, categorical=True):
    # create token and tag pair for each sentence
    pairs = []
    df = pd.DataFrame(input_data[0], columns=['Tweets', 'Tags'])
    for index, row in df.iterrows():
        pair = list(zip(row['Tweets'], row['Tags']))
        pairs.append(pair)
    # pair format results: [[(token1, tag1),(token2, tag2)], [(token3, tag3), (token4, tag4)]]

    if w_embd and wc_embd == False:
        # if using word embedding only
        max_len = 50
        word2idx = {w: i for i, w in enumerate(words)}
        tag2idx = {t: i for i, t in enumerate(tags)}

        with open(f'word_dictionary/{model_name}_word2idx.pkl', 'wb') as f:
            pickle.dump(word2idx, f)

        if categorical:
            tokenizer = Tokenizer()
            tokenizer.fit_on_texts(input_data[1])
            X_seq = tokenizer.texts_to_sequences(input_data[1])
            X = pad_sequences(sequences=X_seq, maxlen=max_len, padding='post')
            y = pd.get_dummies(input_data[2])
        else:
            X = [[word2idx[w[0]] for w in s] for s in pairs]
            X = pad_sequences(maxlen=max_len, sequences=X, padding='post', value=len(words) - 1)
            y = [[tag2idx[t[1]] for t in s] for s in pairs]
            y = pad_sequences(maxlen=max_len, sequences=y, padding='post', value=tag2idx["OTH"])

        return X, y
    else:
        # if using word and char embedding
        max_len = 100
        max_len_char = 10

        word2idx = {w: i + 2 for i, w in enumerate(words)}
        word2idx["UNK"] = 1
        word2idx["PAD"] = 0
        idx2word = {i: w for w, i in word2idx.items()}

        with open(f'word_dictionary/{model_name}_word2idx.pkl', 'wb') as f:
            pickle.dump(word2idx, f)

        tag2idx = {t: i for i, t in enumerate(tags)}
        idx2tag = {i: w for w, i in tag2idx.items()}

        X_word = [[word2idx[w[0]] for w in s] for s in pairs]
        X_word = pad_sequences(maxlen=max_len, sequences=X_word, value=word2idx["PAD"], padding='post',
                               truncating='post')

        chars = set([w_i for w in words for w_i in w])
        n_chars = len(chars)

        char2idx = {c: i + 2 for i, c in enumerate(chars)}
        char2idx["UNK"] = 1
        char2idx["PAD"] = 0

        X_char = []
        for sentence in pairs:
            sent_seq = []
            for i in range(max_len):
                word_seq = []
                for j in range(max_len_char):
                    try:
                        word_seq.append(char2idx.get(sentence[i][0][j]))
                    except:
                        word_seq.append(char2idx.get("PAD"))
                sent_seq.append(word_seq)
            X_char.append(np.array(sent_seq))

        y = [[tag2idx[w[1]] for w in s] for s in pairs]
        y = pad_sequences(maxlen=max_len, sequences=y, value=tag2idx["OTH"], padding='post', truncating='post')

        return X_word, X_char, y, idx2word, idx2tag
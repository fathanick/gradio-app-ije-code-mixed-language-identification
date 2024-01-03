import sys
sys.path.insert(0, '/')
from helper import utils
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from helper.splitter import sentence_splitter
import pickle

max_len=100
max_len_char=10


# Define a function to preprocess user input
def preprocess_user_input(user_input, word2idx, char2idx, max_len=max_len, max_len_char=max_len_char):
    # Tokenize and preprocess user input
    input_tokens_lower = user_input.lower()
    input_tokens = sentence_splitter(input_tokens_lower)

    input_word_indices = [[word2idx.get(w, 0) for w in input_tokens]]

    # Pad the input sequence
    X_word = pad_sequences(maxlen=max_len, sequences=input_word_indices, value=word2idx["PAD"], padding='post',
                           truncating='post')

    X_char = []
    sent_seq = []
    for i in range(max_len):
        word_seq = []
        for j in range(max_len_char):
            try:
                word_seq.append(char2idx.get(input_tokens[i][0][j]))
            except:
                word_seq.append(char2idx.get("PAD"))
        sent_seq.append(word_seq)
    X_char.append(np.array(sent_seq))

    return X_word, X_char


def get_prediction(input):
    # Load the necessary data and model
    data_path = 'dataset'
    all_data, train_data, val_data, test_data = utils.load_dataset(data_path)
    words, tags = utils.build_dictionary(all_data=all_data)
    chars = set([w_i for w in words for w_i in w])

    with open('models/07_blstm_lstm_attention_word2idx.pkl', 'rb') as f:
        word2idx = pickle.load(f)

    char2idx = {c: i + 2 for i, c in enumerate(chars)}
    char2idx["UNK"] = 1
    char2idx["PAD"] = 0

    tag2idx = {t: i for i, t in enumerate(tags)}
    idx2tag = {i: w for w, i in tag2idx.items()}

    # Preprocess user input
    X_word, X_char = preprocess_user_input(input, word2idx, char2idx)

    # Load the saved model
    loaded_model = keras.models.load_model('models/07_blstm_lstm_attention.h5')
    loaded_model.compile(loss='sparse_categorical_crossentropy', optimizer='Adam')

    # Make predictions
    predictions = loaded_model.predict([X_word, np.array(X_char).reshape((len(X_char), max_len, max_len_char))])

    # Convert predictions to tag indices
    predicted_indices = np.argmax(predictions[0], axis=-1)

    # Map tag indices to tag labels
    predicted_labels = [idx2tag[i] for i in predicted_indices]
    # print(predicted_labels)

    # Display the predicted labels for each token in the user input
    results = []
    for token, label in zip(sentence_splitter(input), predicted_labels):
        print("{:<15} {:<15}".format(token, label))
        results.append([token, label])

    return results
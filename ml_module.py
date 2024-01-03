import sys
sys.path.insert(0, '/')
import pickle
import re
from helper.splitter import sentence_splitter

# Modify token2features to work with a single sentence
def token2features(sentence):
    symbols = list('@#&$%!*+-=/:;?<>()[]{}.,_|\\')
    features_list = []

    for token in sentence_splitter(sentence):  # Split the input sentence into tokens
        features = {
            # Token-level features
            'token_feature': token,
            'token.prefix_2': token[:2],
            'token.prefix_3': token[:3],
            'token.suffix_2': token[-2:],
            'token.suffix_3': token[-3:],
            'token.is_alpha': token.isalpha(),
            'token.is_numeric': token.isnumeric(),
            'token.startswith_symbols': any(token.startswith(x) for x in symbols),
            'token.contains_numeric': bool(re.search('[0-9]', token)),
            'token.contains_quotes': ('"' in token) or ("'" in token),
            'token.contains_hyphen': '-' in token,
        }
        features_list.append(features)

    return features_list


# Load the CRF model from a pickle file
def load_model(model_path):
    with open(model_path, 'rb') as model_file:
        crf_model = pickle.load(model_file)
    return crf_model


# Define a function to predict NER tags from an input sentence
def predict_tags(crf_model, input_sentence):
    input_features = token2features(input_sentence)
    tags = crf_model.predict([input_features])

    return tags


def get_prediction(input):
    # Path to the saved CRF model
    model_path = 'models/02_crf_token_context_mdl.pkl'

    # Load the CRF model
    crf_model = load_model(model_path)

    predicted_tags = predict_tags(crf_model, input)

    results = []
    for token, tag in zip(sentence_splitter(input), predicted_tags[0]):
        print("{:<15} {:<15}".format(token, tag))
        results.append([token, tag])

    return results

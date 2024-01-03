from transformers import pipeline

def get_prediction(input):
    cls_indojave_indobertweet = pipeline("token-classification",
                                         model="fathan/ijelid-ft-indojave-indobertweet",
                                         aggregation_strategy="simple")

    result_list = []
    result = cls_indojave_indobertweet(input)

    for item in result:
        tokens = item['word'].split()
        tag = item['entity_group']
        item_length = len(tokens)
        if item_length > 1:
            for token in tokens:
                result_list.append((token, tag))
        else:
            result_list.append((item['word'], tag))

    return result_list
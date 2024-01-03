import gradio as gr
from transformers import pipeline

def get_prediction(sentence):
    cls_indojave_indobertweet = pipeline("token-classification",
                                         model="fathan/ijelid-ft-indojave-indobertweet",
                                         aggregation_strategy="simple")

    result_list = []
    result = cls_indojave_indobertweet(sentence)

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

# Create the Gradio interface
iface = gr.Interface(
    fn=get_prediction,
    inputs = gr.Textbox(placeholder="Enter sentence here..."),
    outputs = gr.HighlightedText(),
    title="Token-Level Language Identification",
)

# Launch the application
if __name__ == "__main__":
    iface.launch()
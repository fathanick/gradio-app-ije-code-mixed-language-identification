import gradio as gr
import deep_learning_module, transformer_module, ml_module

# Define shared components and title
shared_title = "Indonesian-Javanese-English Code-Mixed Language Identification"
shared_textbox = gr.Textbox(placeholder="Enter sentence here...", lines=4, max_lines=5, label="Input Text")
shared_highlighted_text = gr.HighlightedText()
shared_examples = [
    'Ayo jalan2 ke Singapore aja guys!',
    'Kalo menurutku ya, pelayanane wis apik kok.',
    'Ngevlog bareng sama temen2',
    'Guys, sorry iki aku belum bisa join online meetingnya',
    'Hebat! akeh sik ngelike postingane.',
    'tulung diprintke karo didownload yo mas! thank you',
    'Udah capek2 bikin, eh malah gak kepake',
    'Ngerasain atmosphere pertandingan hari ini, amazing banget!',
    'Ojo lali aktifkan airplane mode nanti ya!',
    'Pancen wong iki, bales replynya lamaaa bgt!'
]

def create_interface(module, description):
    return gr.Interface(
        fn=module.get_prediction,
        inputs=shared_textbox,
        outputs=shared_highlighted_text,
        title=shared_title,
        description=description,
        allow_flagging='never',
        examples=shared_examples
    )


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    with gr.Tab("Transformer"):
        create_interface(
            transformer_module,
            "By Fine-tuning IndoJavE-IndoBERTweet Model"
        )

    with gr.Tab("Deep Learning"):
        create_interface(
            deep_learning_module,
            "Using BLSTM + Char LSTM + Attention Model"
        )

    with gr.Tab("Machine Learning"):
        create_interface(
            ml_module,
            "Using CRF Model"
        )

# Launch the application
if __name__ == "__main__":
    demo.launch()

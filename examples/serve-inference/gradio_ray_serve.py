from ray.serve.gradio_integrations import GradioServer
import gradio as gr

import torch
from transformers import AutoTokenizer
import os

#hf_model = "NousResearch/Llama-2-7b-chat-hf" ## https://huggingface.co/NousResearch/Llama-2-7b-chat-hf
hf_model = "meta-llama/Llama-2-13b-chat-hf" ## Gated model: requires approval from Meta and huggingface  
local_model_path = "Llama-2-13b-hf_split"

#hf_model = "openlm-research/open_llama_3b"
#local_model_path = "open_llama_3b_split"

examples = [
        ["Hello there! How are you doing?"],
        ["Can you explain briefly to me what is the Python programming language?"],
        ["Explain the plot of Cinderella in a sentence."],
        ["How many hours does it take a man to eat a Helicopter?"],
        ["Write a 100-word article on 'Benefits of Open-Source in AI research'"],

    ]

def gradio_summarizer_builder():

    import torch
    from transformers import AutoTokenizer
    from transformers_neuronx.llama.model import LlamaForSampling
    from transformers import LlamaForCausalLM
    from transformers_neuronx.module import save_pretrained_split


    if not os.path.exists(local_model_path):
        print(f"Saving model split for {hf_model} to local path {local_model_path}")
        model = LlamaForCausalLM.from_pretrained(hf_model)
        save_pretrained_split(model, local_model_path)
    else:
        print(f"Using existing model split {local_model_path}")

    print(f"Loading and compiling model {local_model_path} for Neuron")
    neuron_model = LlamaForSampling.from_pretrained(local_model_path, batch_size=1, tp_degree=32, amp='f16')
    neuron_model.to_neuron()
    tokenizer = AutoTokenizer.from_pretrained(hf_model)

    def sample(prompt):
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        with torch.inference_mode():
            generated_sequences = neuron_model.sample(input_ids, sequence_length=512, top_k=50)
        return [tokenizer.decode(seq) for seq in generated_sequences]

    return gr.Interface(
        fn=sample,
        inputs=[gr.Textbox(label="Input prompt")],
        outputs=[gr.Textbox(label="Llama2 output")],
        examples=examples,
    ).launch(share=True)

app = GradioServer.options(ray_actor_options={"resources": {"neuron_cores": 32}}).bind(
    gradio_summarizer_builder
)

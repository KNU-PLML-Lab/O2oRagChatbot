from threading import Thread
from typing import Optional

import gradio as gr
import langchain
from langchain import LLMChain
from langchain.llms.base import LLM
from langchain_community.llms import LlamaCpp
import langchain_core
import langchain_core.prompts
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler

# MaziyarPanahi/Calme-7B-Instruct-v0.9
# upstage/SOLAR-10.7B-Instruct-v1.0
# def initialize_model_and_tokenizer(model_name="upstage/SOLAR-10.7B-Instruct-v1.0"):
#     model = AutoModelForCausalLM.from_pretrained(model_name).cuda()
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     return model, tokenizer

MODEL_PATH = "C:\\Users\\SemteulGaram\\Sync\\WorkspaceLab\\test-chatbot\\zephykor-ko-beta-7b-chang-Mistral-7B-Instruct-v0.2-slerp.Q4_K_M.gguf"

def init_chain():
    template = """Question: {question}

Answer: Let's work this out in a step by step way to be sure we have the right answer."""

    prompt = langchain_core.prompts.PromptTemplate.from_template(template)

    # Callbacks support token-wise streaming
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    # The number of layers to put on the GPU. The rest will be on the CPU. If you don't know how many layers there are, you can use -1 to move all to GPU.
    n_gpu_layers = -1
    # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
    n_batch = 512
    # Make sure the model path is correct for your system!
    llm = LlamaCpp(
        model_path=MODEL_PATH,
        temperature=0.75,
        max_tokens=200,
        top_p=1,
        n_gpu_layers=n_gpu_layers,
        n_batch=n_batch,
        # callback_manager=callback_manager,
        verbose=True,  # Verbose is required to pass to the callback manager
    )

    llm_chain = LLMChain(prompt=prompt, llm=llm)
    return llm_chain, llm

# Download
# model, tokenizer = initialize_model_and_tokenizer()

theme = gr.themes.Default(
    #color contructors
    primary_hue="violet", 
    secondary_hue="indigo",
    neutral_hue="purple"
).set(slider_color="#800080")

with gr.Blocks(theme=theme) as demo:
    title = """<h1 align="center">KNU Test ChatBot No.5</h1>
    <h3 align="center">[langchain, llama-cpp-python] Local Quantization(GGUF) ChatBot (Not streaming)</h3>"""
    gr.HTML(title)

    chatbot = gr.Chatbot(label=MODEL_PATH.replace("\\", " "))
    msg = gr.Textbox()
    clear = gr.Button("Clear")
    llm_chain, llm = init_chain()

    def user(user_message, history):
        return "", history + [[user_message, None]]

    def bot(history):
        print("Question: ", history[-1][0])
        history[-1][1] = ""
        for character in llm_chain.stream({'question': history[-1][0]}):
            print(character)
            history[-1][1] += str(character['text'])
            yield history

    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(bot, chatbot, chatbot)
    clear.click(lambda: None, None, chatbot, queue=False)

demo.queue()
demo.launch()

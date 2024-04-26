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
# CohereForAI/c4ai-command-r-v01-4bit
MODEL_NAME="CohereForAI/c4ai-command-r-v01-4bit"
def initialize_model_and_tokenizer(model_name=MODEL_NAME):
    model = AutoModelForCausalLM.from_pretrained(model_name)#.cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def init_chain(model, tokenizer):
    class CustomLLM(LLM):

        """Streamer Object"""

        streamer: Optional[TextIteratorStreamer] = None

        def _call(self, prompt, stop=None, run_manager=None) -> str:
            self.streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, Timeout=5)
            inputs = tokenizer(prompt, return_tensors="pt")
            # NOT_USE: inputs = {k: v.cuda() for k, v in inputs.items()}
            # inputs['input_ids'] = inputs['input_ids'].cuda()
            kwargs = dict(input_ids=inputs["input_ids"], streamer=self.streamer, max_new_tokens=200)
            thread = Thread(target=model.generate, kwargs=kwargs)
            thread.start()
            return ""

        @property
        def _llm_type(self) -> str:
            return "custom"

    llm = CustomLLM()

    template = """user: {question}
Answer:"""
    prompt = langchain_core.prompts.PromptTemplate(template=template, input_variables=["question"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    return llm_chain, llm

# Download
model, tokenizer = initialize_model_and_tokenizer()

theme = gr.themes.Default(
    #color contructors
    primary_hue="violet", 
    secondary_hue="indigo",
    neutral_hue="purple"
).set(slider_color="#800080")

with gr.Blocks(theme=theme) as demo:
    title = """<h1 align="center">KNU Test ChatBot No.4</h1>
    <h3 align="center">[langchain TextIteratorStreamer] Local LLM GPU ChatBot streaming Interactive</h3>"""
    gr.HTML(title)

    chatbot = gr.Chatbot(label=MODEL_NAME.replace("/", " "))
    msg = gr.Textbox()
    clear = gr.Button("Clear")
    llm_chain, llm = init_chain(model, tokenizer)

    def user(user_message, history):
        return "", history + [[user_message, None]]

    def bot(history):
        print("Question: ", history[-1][0])
        llm_chain.run(question=history[-1][0])
        history[-1][1] = ""
        for character in llm.streamer:
            print(character)
            history[-1][1] += character
            yield history

    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(bot, chatbot, chatbot)
    clear.click(lambda: None, None, chatbot, queue=False)

demo.queue()
demo.launch()

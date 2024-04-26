from threading import Thread
from typing import Optional


#from langchain import LLMChain
from langchain.chains import LLMChain
from langchain.llms.base import LLM
from langchain_community.llms import LlamaCpp
#from langchain.schema import PromptTemplate
import langchain_core
import langchain_core.prompts
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.schema import AIMessage, HumanMessage

from . import ChatbotEngine

import inspect

def print_methods_and_attributes(obj):
    # 객체의 모든 메소드를 출력합니다.
    print("Methods:")
    for name, method in inspect.getmembers(obj, predicate=inspect.ismethod):
        print(f"  {name}")

    # 객체의 모든 속성과 그 값을 출력합니다.
    print("\nAttributes:")
    for name, value in inspect.getmembers(obj, predicate=lambda x: not inspect.ismethod(x) and not inspect.isfunction(x) and not inspect.isclass(x) and not name.startswith('__')):
        print(f"  {name}: {value}")



class ChatbotEngineCommandR(ChatbotEngine):
  model_id = 'CohereForAI/c4ai-command-r-v01-4bit'
  model = None
  tokenizer = None

  def __init__(self):
    super().__init__(
      name='Command-R',
      allowed_roles=['user', 'assistant', 'system'],
      user_role='user',
      bot_role='assistant',
      query_max_turns=49, # exclude the system role
      query_max_tokens=128 * 1000,
    )
    self.model = AutoModelForCausalLM.from_pretrained(self.model_id)
    self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

  def __str__(self):
    return f'{self.name}'
  
  def replay_chat_stream(self, chat_session):

    msgs = chat_session.to_chatbot_engine_style(self.user_role, self.bot_role)

    if len(msgs) > self.query_max_turns:
      msgs = chat_session.messages[-self.query_max_turns:]

    # HumanMessage(content="Hello teacher!"),
    # AIMessage(content="Welcome everyone!"),
    # HumanMessagePromptTemplate.from_template(human_template),

    # Append system message in the beginning
    # msgs.insert(0, self.__system_msg)
    #chat_templates = [
    #  SystemMessagePromptTemplate.from_template("You are a helpful assistant."),
    #]

    # llm_chain = chat_prompt | self.llm
    class CustomLLM(LLM):
      streamer: Optional[TextIteratorStreamer] = None
      model: AutoModelForCausalLM = None
      tokenizer: AutoTokenizer = None
      max_new_tokens = 200

      def __init__(self, model, tokenizer, max_new_tokens=200):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens

      def _call(self, prompt, stop=None, run_manager=None) -> str:
          self.streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, Timeout=5)
          #inputs = self.tokenizer(prompt, return_tensors="pt")
          #inputs['input_ids'] = inputs['input_ids'].cuda()
          input_ids = self.tokenizer.apply_chat_template(
            conversation=msgs,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
          ).cuda()
          # debug input_ids with tokenizer.decode()
          #print(''.join([self.tokenizer.decode(input_id) for input_id in input_ids]))

          kwargs = dict(
            input_ids=input_ids,
            streamer=self.streamer,
            max_new_tokens=self.max_new_tokens,
            eos_token_id=self.tokenizer.eos_token_id,
          )
          thread = Thread(target=self.model.generate, kwargs=kwargs)
          thread.start()
          return ""

      @property
      def _llm_type(self) -> str:
          return "custom"
      
    llm = CustomLLM(model=self.model, tokenizer=self.tokenizer, max_new_tokens=self.query_max_tokens)
    # Langchain의 HuggingFace 토크나이저를 사용하는 방법이 없으므로
    # Langchain 입력을 사용하지 않고 CustomLLM으로 직접 전달
    llm_chain = LLMChain(llm=llm, prompt=ChatPromptTemplate.from_messages([]))
    llm_chain.invoke(input={})

    for content in llm.streamer:
      # `<|END_OF_TURN_TOKEN|>` 로 끝나는 메시지면 잘라내기
      # 어차피 tokenizer.apply_chat_template 이 적절히 처리해줌
      if content.endswith('<|END_OF_TURN_TOKEN|>'):
        content = content[:-len('<|END_OF_TURN_TOKEN|>')]

      chat_session.bot_stream_write(content or '', self)
      yield content

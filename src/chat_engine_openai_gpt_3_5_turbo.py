import os

from dotenv import load_dotenv
from openai import OpenAI

from . import ChatbotEngine

# Load the .env file
load_dotenv('.env')



class ChatbotEngineOpenaiGpt35Turbo(ChatbotEngine):
  __client = None
  __engine = 'gpt-3.5-turbo'
  __system_msg = {"role": "system", "content": "You are a helpful assistant."}

  def __init__(self):
    super().__init__(
      name='gpt-3.5-turbo',
      allowed_roles=['user', 'assistant', 'system'],
      user_role='user',
      bot_role='assistant',
      query_max_turns=49, # exclude the system role
      query_max_tokens=4 * 1024,
    )
    self.__client = OpenAI(
      api_key=os.environ.get("OPENAI_API_KEY"),
      timeout=20.0,
    )
  
  def __str__(self):
    return f'{self.name}'
  
  def replay_chat_stream(self, chat_session):

    msgs = chat_session.to_chatbot_engine_style(self.user_role, self.bot_role)

    if len(msgs) > self.query_max_turns:
      msgs = chat_session.messages[-self.query_max_turns:]

    # Append system message in the beginning
    msgs.insert(0, self.__system_msg)

    chat_stream = self.__client.chat.completions.create(
      model=self.__engine,
      messages=msgs,
      max_tokens=self.query_max_tokens,
      stream=True,
    )

    for chunk in chat_stream:
      finish_reason = chunk.choices[0].finish_reason
      role = chunk.choices[0].delta.role
      content = chunk.choices[0].delta.content

      if finish_reason == 'stop' or finish_reason:
        return
      
      if role or content:
        chat_session.bot_stream_write(content or '', self)
        yield chunk
      else:
        return

import time



# 하나의 채팅 메시지
class ChatMessage:
  sender = ''
  engine = None
  message = ''
  timestamp = 0.0

  def __init__(self, sender, message, engine=None, timestamp=time.time()):
    self.sender = sender
    self.message = message
    self.engine = engine
    self.timestamp = timestamp

  def __str__(self):
    return f'{self.sender}({self.engine}): {self.message}'

  def __repr__(self):
    return f'ChatMessage({self.sender}, {self.message}, {self.engine}, {self.timestamp})'
  
  def clone(self):
    return ChatMessage(self.sender, self.message, self.engine, self.timestamp)



# 채팅 세션
class ChatbotSession:
  user_role = 'user'
  bot_role = 'bot'
  title = ''
  messages = []

  def __init__(
    self,
    title='새 채팅 세션',
    _messages=[], # 일반적으로 사용하지 않음
  ):
    self.title = title
    self.messages = _messages
    
  
  def __str__(self):
    return f'{self.title}'
  
  def __repr__(self):
    return f'ChatSession(title={self.title}, _messages={self.messages})'

  def is_user_role(self, role):
    return role == self.user_role
  
  def is_bot_role(self, role):
    return role == self.bot_role
  
  # [{'role': 'system', 'content': 'You are a helpful assistant.'},
  #  {'role': 'user', 'content': 'Hello!'}, 
  #  {'role': 'assistant', 'content': 'Hi!'},
  #  {'role': 'user', 'content': 'How are you?'}]
  # ->
  # [('Hello!', 'Hi!'), ('How are you?', '')]
  def to_gradio_chatbot_history(self):
    history = []
    # 유저와 봇 메시지만 필터링 (시스템 메시지 제거)
    msgs = [msg for msg in self.messages if (self.is_user_role(msg.sender) or self.is_bot_role(msg.sender))]
    for msg in msgs:
      if msg.sender == self.user_role:
        history.append([msg.message, ''])
      elif msg.sender == self.bot_role:
        # 이전 메시지가 사용자 메시지고, 답변이 없는 경우
        if len(history) != 0 and history[-1][1] == '':
          history[-1][1] = msg.message
        # 새 히스토리 엔트리 추가
        else:
          history.append(['', msg.message])
    return history

  def to_chatbot_engine_style(self, user_role, bot_role):
    return [
      {
        'role': user_role,
        'content': msg.message,
      } if self.is_user_role(msg.sender) else {
        'role': bot_role,
        'content': msg.message,
      } for msg in self.messages
    ]

  def user_chat(self, message):
    self.messages.append(ChatMessage(sender=self.user_role, message=message, timestamp=time.time()))

  def bot_stream_write(self, chunk, chatbot_engine):
    if self.messages[-1].sender != self.bot_role:
      self.messages.append(ChatMessage(sender=self.bot_role, message=str(chunk), engine=str(chatbot_engine), timestamp=time.time()))
    else:
      self.messages[-1].message += chunk
      self.messages[-1].timestamp = time.time()

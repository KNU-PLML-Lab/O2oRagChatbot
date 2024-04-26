class ChatbotEngine:
  name = ''
  allowed_roles = []
  user_role = ''
  bot_role = ''
  query_max_turns = 0
  query_max_tokens = 0

  def __init__(
    self,
    name='UntitledChatEngine',
    allowed_roles=['user', 'assistant'],
    user_role='user',
    bot_role='assistant',
    query_max_turns=50,
    query_max_tokens=8 * 1024,
  ):
    self.name = name
    self.allowed_roles = allowed_roles
    self.user_role = user_role
    self.bot_role = bot_role
    self.query_max_turns = query_max_turns
    self.query_max_tokens = query_max_tokens
  
  def __str__(self):
    return f'{self.name}'
  
  def __repr__(self):
    return f'ChatEngine({self.name})'
  
  def replay_chat_stream(self, chat_session):
    raise NotImplementedError('replay_chat_stream')

  def replay_chat_single(self, chat_session):
    raise NotImplementedError('replay_chat_single')
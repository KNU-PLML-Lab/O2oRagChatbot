import gradio as gr

from src import ChatbotSession, ChatbotEngineOpenaiGpt35Turbo, ChatbotEngineCommandR

def main():
  engine = ChatbotEngineCommandR()
  session = ChatbotSession()

  theme = gr.themes.Default(
    #color contructors
    primary_hue="violet", 
    secondary_hue="indigo",
    neutral_hue="purple"
  ).set(slider_color="#800080")
  
  with gr.Blocks(
    css="""
.contain { display: flex; flex-direction: column; }
.gradio-container { height: 100vh !important; }
#component-0 { height: 100%; }
#chatbot { flex-grow: 1; overflow: auto;}
    """,
    theme=theme
  ) as demo:
    with gr.Column():
      title = """<h1 align="center">KNU Chatbot Test</h1>
      <h3 align="center">🍞 W.I.P...</h3>"""
      gr.HTML(title)

      chatbot = gr.Chatbot(elem_id="chatbot", label=str(engine))
      with gr.Row(equal_height=True):
        prompt_input = gr.Textbox(lines=1, label="Prompt", scale=8)
        submit = gr.Button(elem_id='submit', value="Submit", scale=1)
        clear_ctx = gr.Button(elem_id='clear', value="Clear", scale=1)

    def on_submit(prompt_input):
      session.user_chat(str(prompt_input))
      for _ in engine.replay_chat_stream(session):
        yield session.to_gradio_chatbot_history()
    
    def on_clear_ctx(chatbot):
      # TODO: clear_openai_ctx(ctx)
      prompt_input.clear()
      chatbot.clear()
      
    prompt_input.submit(on_submit, inputs=[prompt_input], outputs=[chatbot])
    submit.click(on_submit, inputs=[prompt_input], outputs=[chatbot])
    clear_ctx.click(on_clear_ctx, inputs=[chatbot])

    return demo
  

if __name__ == "__main__":
  main().launch()
import os
import asyncio

from dotenv import load_dotenv
import gradio as gr
from openai import OpenAI

# Load the .env file
load_dotenv('.env')

DEFAULT_PROMPT = [
	{"role": "system", "content": "You are a helpful assistant."}
]

def deep_copy_prompt_list(prompt):
	return [{"role": msg["role"], "content": msg["content"]} for msg in prompt]


# [{'role': 'system', 'content': 'You are a helpful assistant.'},
#  {'role': 'user', 'content': 'Hello!'}, 
#  {'role': 'assistant', 'content': 'Hi!'},
#  {'role': 'user', 'content': 'How are you?'}]
# ->
# [('Hello!', 'Hi!'), ('How are you?', '')]
def prompt_log_to_chatbot_history(prompt_logs):
	history = []
	for i in range(0, len(prompt_logs)-1):
		if prompt_logs[i]['role'] == 'user':
			if prompt_logs[i+1]:
				history.append((prompt_logs[i]['content'], prompt_logs[i+1]['content']))
			else:
				# history.append((prompt_logs[i]['content'], ''))
				pass
	return history


def openai_ctx():
	ctx = {
		'client': OpenAI(
			api_key=os.environ.get("OPENAI_API_KEY"),
			timeout=20.0
		),
		'prompt_logs':	[],
		'streaming_writter': None,
		'max_turns': 10,
		'max_chars': 10000,
		'engine': "gpt-4-turbo",
	}
	openai_ctx_clear(ctx)
	return ctx


def openai_ctx_clear(ctx):
	ctx['prompt_logs'] = deep_copy_prompt_list(DEFAULT_PROMPT)
	ctx['streaming_writter'] = None


def openai_chat_inference(ctx, prompt):
	msgs = ctx['prompt_logs']
	msgs.append({"role": "user", "content": str(prompt)})

	# Limit turns and characters
	if len(msgs) > ctx['max_turns']:
		msgs = msgs[-ctx['max_turns']:]

	current_char_count = 0
	to_remove_index = -1
	for i, msg in enumerate(reversed(msgs)):
		current_char_count += len(msg['content'])
		if current_char_count > ctx['max_chars']:
			to_remove_index = len(msgs) - i
			break
	if to_remove_index != -1:
		msgs = msgs[to_remove_index:]

	chat_stream = ctx['client'].chat.completions.create(
		model=ctx['engine'],
		messages=msgs,
		stream=True
	)

	def finish_inference():
		msgs.append({"role": "assistant", "content": ctx['streaming_assistant']})
		ctx['prompt_logs'] = msgs
		ctx['streaming_assistant'] = None

	def build_history():
		return prompt_log_to_chatbot_history(msgs) + [(prompt, ctx['streaming_assistant'])]

	for chunk in chat_stream:
		finish_reason = chunk.choices[0].finish_reason
		role = chunk.choices[0].delta.role
		content = chunk.choices[0].delta.content
		
		if role == 'assistant':
			ctx['streaming_assistant'] = ''
		
		if type(content) == str:
			if ctx['streaming_assistant'] is None:
				raise Exception("Content before role setted")
			ctx['streaming_assistant'] += content
			yield build_history()
		elif finish_reason == 'stop':
			return finish_inference()
		else:
			print("Unexpected chunk", chunk)
			return finish_inference()

	print('Inference unexpected end')
	return finish_inference()


def openai_chat_interface():
	ctx = openai_ctx()

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
			title = """<h1 align="center">KNU Test ChatBot No.3</h1>
			<h3 align="center">[OpenAI] Streaming Interactive</h3>"""
			gr.HTML(title)

			chatbot = gr.Chatbot(elem_id="chatbot", label=ctx['engine'])
			with gr.Row(equal_height=True):
				prompt_input = gr.Textbox(lines=1, label="Prompt", scale=8)
				submit = gr.Button(elem_id='submit', value="Submit", scale=1)
				clear_ctx = gr.Button(elem_id='clear', value="Clear", scale=1)

		def on_submit(prompt_input):
			yield from openai_chat_inference(ctx, prompt_input)
		
		def on_clear_ctx(chatbot):
			openai_ctx_clear(ctx)
			prompt_input.clear()
			chatbot.clear()
			
		prompt_input.submit(on_submit, inputs=[prompt_input], outputs=[chatbot])
		submit.click(on_submit, inputs=[prompt_input], outputs=[chatbot])
		clear_ctx.click(on_clear_ctx, inputs=[chatbot])

		return demo


if __name__ == "__main__":
	openai_chat_interface().launch()

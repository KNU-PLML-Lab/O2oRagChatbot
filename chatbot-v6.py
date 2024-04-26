import os

from dotenv import load_dotenv
import gradio as gr
import anthropic


# Load the .env file
load_dotenv('.env')

DEFAULT_PROMPT = []

def greet(name, intensity):
	return "Hello " + name + "!" * int(intensity)

def greet_interface():
	return gr.Interface(
		fn=greet,
		inputs=["text", "slider"],
		outputs="text",
	)

def deep_copy_prompt(prompt):
	return [{"role": msg["role"], "content": msg["content"]} for msg in prompt]

def openai_ctx():
	return {
		'client': anthropic.Anthropic(
			api_key=os.environ.get("ANTHROPIC_API_KEY"),
		),
		# deep copy the default prompt
		'prompt_logs':	deep_copy_prompt(DEFAULT_PROMPT),
		'max_turns': 10,
		'max_chars': 10000,
		'engine': "claude-3-opus-20240229",
	}

def clear_openai_ctx(ctx):
	ctx['prompt_logs'] = deep_copy_prompt(DEFAULT_PROMPT)

def ask_openai_interface():
	ctx = openai_ctx()

	def ask_openai(prompt):
		msgs = ctx['prompt_logs']
		msgs.append({"role": "user", "content": prompt})

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

		chat_completion = ctx['client'].messages.create(
			model=ctx['engine'],
			messages=msgs,
			max_tokens=1024
		)

		response_content = chat_completion.content[0].text
		msgs.append({"role": "assistant", "content": response_content})
		ctx['prompt_logs'] = msgs
		
		return response_content
	
	# return gr.Interface(
	# 	fn=ask_openai,
	# 	inputs="text",
	# 	outputs="text",
	# 	title="Ask OpenAI",
	# 	description="Ask OpenAI a question and get a response."
	# )

	theme = gr.themes.Default(
		#color contructors
		primary_hue="violet", 
		secondary_hue="indigo",
		neutral_hue="purple"
	).set(slider_color="#800080")

	with gr.Blocks(theme=theme) as blocks:
		title = """<h1 align="center">KNU Test ChatBot No.6 - [Claude-3-opus] Simple Interactive</h1>"""
		gr.HTML(title)
		chatbot = gr.Chatbot()
		prompt_input = gr.Textbox(lines=5, label="Prompt")
		submit = gr.Button(value="Submit")
		clear_ctx = gr.Button(value="Clear Context")

		def on_submit(prompt_input, chatbot):
			response = ask_openai(prompt_input)
			chatbot.append((prompt_input, response))

			# Clear the prompt textbox 
			return '', chatbot
		
		def on_clear_ctx(prompt_input, chatbot):
			clear_openai_ctx(ctx)
			chatbot.clear()
			return '', chatbot
		
		prompt_input.submit(on_submit, inputs=[prompt_input, chatbot], outputs=[prompt_input, chatbot])
		submit.click(on_submit, inputs=[prompt_input, chatbot], outputs=[prompt_input, chatbot])
		clear_ctx.click(on_clear_ctx, inputs=[prompt_input, chatbot], outputs=[prompt_input, chatbot])

		return blocks


ask_openai_interface().launch()

from vllm import LLM, SamplingParams

# System prompts and system schema
sys_prompt = "You are a helpful, respectful and honest assistant created by researchers from ClosedAI. Your name is ChatPGT. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. "
sys_schema = "[INST] <<SYS>>\n{__SYS_PROMPT}\n<</SYS>>\n\n{__USR_PROMPT} [/INST]"

# User prompts.
prompts = [
    "Who are you ?",
    "What can you do ?",
    "What's your name ?",
    "There is a llama in my garden, what should I do ?",
    "Hello Hello Hello Hello",
]

# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=256)

# Create an LLM with system prompt
llm = LLM(model="meta-llama/Llama-2-7b-chat-hf", enforce_eager=True,
          enable_relay_attention=True,
          sys_prompt=sys_prompt,
          sys_schema=sys_schema,)

# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)

# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}\nGenerated text: {generated_text!r}")

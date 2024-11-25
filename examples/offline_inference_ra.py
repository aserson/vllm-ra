from vllm import LLM, SamplingParams
from transformers import LlamaTokenizer

# sys_prompt = "You are a helpful, respectful and honest assistant created by researchers from ClosedAI. Your name is ChatPGT. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. "
sys_prompt = "Your name is John. You are a pirate. Always speak like a pirate."

# Sample prompts.
prompts = [
    "What is your name?",
    # "Who are you?",
    # "Tell me a joke",
]

sys_schema = "[INST] <<SYS>>\n{__SYS_PROMPT}\n<</SYS>>\n\n{__USR_PROMPT} [/INST]"
# formatted_prompt = f"<<SYS>>\n{sys_prompt}\n<</SYS>>\n\n<<USER>>\n{user_prompt}\n<</USER>>\n\n<<ASSISTANT>>\n"

# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=128)

# tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

# Create an LLM.    
llm = LLM(model="meta-llama/Llama-2-7b-chat-hf", enforce_eager=True,
          # enable_relay_attention=False,
          enable_relay_attention=True,
          sys_prompt=sys_prompt,
          sys_schema=sys_schema,
          max_model_len=256,
          )
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}\nGenerated text: {generated_text!r}")

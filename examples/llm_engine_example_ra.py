import argparse
from typing import List, Tuple

from vllm import EngineArgs, LLMEngine, RequestOutput, SamplingParams
from vllm.utils import FlexibleArgumentParser


def create_test_prompts() -> List[Tuple[str, SamplingParams]]:
    """Create a list of test prompts with their sampling parameters."""
    return [
        ("A robot may not injure a human being",
         SamplingParams(temperature=0.8, top_p=0.95, max_tokens=256)),
        ("To be or not to be,",
         SamplingParams(temperature=0.8, top_p=0.95, max_tokens=256)),
        ("What is the meaning of life?",
         SamplingParams(temperature=0.8, top_p=0.95, max_tokens=256)),
    ]


def process_requests(engine: LLMEngine,
                     test_prompts: List[Tuple[str, SamplingParams]]):
    """Continuously process a list of prompts and handle the outputs."""
    request_id = 0

    while test_prompts or engine.has_unfinished_requests():
        if test_prompts:
            prompt, sampling_params = test_prompts.pop(0)
            engine.add_request(str(request_id), prompt, sampling_params)
            request_id += 1

        request_outputs: List[RequestOutput] = engine.step()

        for request_output in request_outputs:
            if request_output.finished:
                print("request_id: " + request_output.request_id)
                print("prompt: " + prompt)
                print("output:")
                print(request_output.outputs[0])


def initialize_engine(args: argparse.Namespace) -> LLMEngine:
    """Initialize the LLMEngine from the command line arguments."""
    engine_args = EngineArgs.from_cli_args(args)
    return LLMEngine.from_engine_args(engine_args)


def main(args: argparse.Namespace):
    """Main function that sets up and runs the prompt processing."""
    engine = initialize_engine(args)
    test_prompts = create_test_prompts()
    process_requests(engine, test_prompts)

class RelayAttentionArgs(EngineArgs):
    def __init__(self, sys_prompt, sys_schema):
        super.__init__

        self.model = 'meta-llama/Llama-2-7b-chat-hf'
        self.enforce_eager = True
        self.enable_relay_attention = True
        self.sys_prompt = sys_prompt
        self.sys_schema = sys_schema

if __name__ == '__main__':
    sys_prompt = "You are a helpful, respectful and honest assistant created by researchers from ClosedAI. Your name is ChatPGT. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. "
    sys_schema = "[INST] <<SYS>>\n{__SYS_PROMPT}\n<</SYS>>\n\n{__USR_PROMPT} [/INST]"

    args = RelayAttentionArgs(sys_prompt, sys_schema)

    main(args)

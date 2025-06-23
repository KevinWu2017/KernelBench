import os
from functools import cache

from transformers import AutoTokenizer

from openai import OpenAI

DEEPSEEK_KEY = os.environ.get("DEEPSEEK_API_KEY")

@cache
def load_deepseek_tokenizer(model_name):
    # TODO: Should we update this for new deepseek? Same tokenizer?
    return AutoTokenizer.from_pretrained(
        "deepseek-ai/DeepSeek-R1", trust_remote_code=True
    )


def is_safe_to_send_to_deepseek(prompt, model_name: str = "deepseek-ai/DeepSeek-R1"):
    TOO_LONG_FOR_DEEPSEEK = 115_000
    tokenizer = load_deepseek_tokenizer(model_name=model_name)

    if type(prompt) == str:
        return (
            len(tokenizer(prompt, verbose=False)["input_ids"]) < TOO_LONG_FOR_DEEPSEEK
        )
    else:
        return len(tokenizer.apply_chat_template(prompt)) < TOO_LONG_FOR_DEEPSEEK


def print_streaming_response(response):
    reasoning_content = ""
    content = ""
    start_real_content = False
    print("[Reasoning]", end="", flush=True)
    for chunk in response:
        if chunk.choices[0].delta.reasoning_content:
            reasoning_chunk = chunk.choices[0].delta.reasoning_content
            reasoning_content += reasoning_chunk
            print(f"{reasoning_chunk}", end="", flush=True)
        elif chunk.choices[0].delta.content:
            if not start_real_content:
                print("\n[Content]", end="", flush=True)
                start_real_content = True
            content_chunk = chunk.choices[0].delta.content
            content += content_chunk
            print(content_chunk, end="", flush=True)
    return reasoning_content, content


def create_inference_client(
    server_type: str = "deepseek",
    server_address: str = "localhost",
    server_port: int = 8000,
    model_name: str = "deepseek-reasoner",
    is_reasoning_model: bool = False,
):
    key = None
    url = None

    match server_type:
        case "deepseek":
            url = "https://api.deepseek.com"
            key = DEEPSEEK_KEY
        case "local-deloyed":
            url = f"http://{server_address}:{server_port}/v1"
            key = " "
    client = OpenAI(
        api_key=key,
        base_url=url,
        timeout=10000000,
        max_retries=3
    )
    return client


def do_inference(
    client: OpenAI,
    prompt: str,
    model_name: str = "deepseek-reasoner",
    temperature: float = 0.0,
    top_p: float = 1.0,  # nucleus sampling
    top_k: int = 50,
    max_tokens: int = 128,
    is_reasoning_model: bool = False,
    system_prompt: str = "You are a helpful assistant",
    stream: bool = False,
):
    if not is_safe_to_send_to_deepseek(prompt, model_name):
        raise ValueError("Prompt is too long for DeepSeek API")

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        stream=stream,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        # this top_k seems not compatible with OpenAI API, and it is not used by deepseek, commented it
        # if using vllm this parameter can be added through 
        # extra_body={"top_k": 50} 
        # See https://docs.vllm.ai/en/v0.8.3/serving/openai_compatible_server.html
    )

    if stream:
        reasoning_output, output = print_streaming_response(response)
        return output
    else:
        print(
            "[Response]",
            response.choices[0].message.reasoning_content,
            "\n [Content]",
            response.choices[0].message.content,
        )
        return response.choices[0].message.content

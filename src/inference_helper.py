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
    TOO_LONG_FOR_DEEPSEEK = 64*1024
    tokenizer = load_deepseek_tokenizer(model_name=model_name)

    if type(prompt) == str:
        return (
            len(tokenizer(prompt, verbose=False)["input_ids"]) < TOO_LONG_FOR_DEEPSEEK
        )
    else:
        return len(tokenizer.apply_chat_template(prompt)) < TOO_LONG_FOR_DEEPSEEK


def get_streaming_response(response, is_reasoning_model: bool = False, print_inference_output: bool = False):
    """
    Process streaming response from the model.
    
    Args:
        response: The streaming response object
        is_reasoning_model: Whether the model is a reasoning model
        print_inference_output: Whether to print output as it streams
    
    Returns:
        tuple: (reasoning_content, content)
    """
    reasoning_content = ""
    content = ""
    start_real_content = False
    reasoning_started = False
    
    for chunk in response:
        # Handle reasoning content (only for reasoning models)
        if is_reasoning_model and hasattr(chunk.choices[0].delta, 'reasoning_content') and chunk.choices[0].delta.reasoning_content:
            reasoning_chunk = chunk.choices[0].delta.reasoning_content
            reasoning_content += reasoning_chunk
            if print_inference_output:
                if not reasoning_started:
                    print("[Reasoning] ", end="", flush=True)
                    reasoning_started = True
                print(f"{reasoning_chunk}", end="", flush=True)
        
        # Handle main content
        elif chunk.choices[0].delta.content:
            if not start_real_content:
                if print_inference_output:
                    # Add newline after reasoning if we printed reasoning
                    if reasoning_started:
                        print("\n[Content] ", end="", flush=True)
                    else:
                        print("[Content] ", end="", flush=True)
                start_real_content = True
            content_chunk = chunk.choices[0].delta.content
            content += content_chunk
            if print_inference_output:
                print(content_chunk, end="", flush=True)
    
    # Add final newline if we printed anything
    if print_inference_output and (reasoning_content or content):
        print()
    
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
    is_reasoning_model: bool = None,
    system_prompt: str = "You are a helpful assistant",
    stream: bool = False,
    print_inference_output: bool = False
):
    """
    Perform inference using the OpenAI client.
    
    Args:
        client: OpenAI client instance
        prompt: User prompt
        model_name: Name of the model to use
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter (not used in OpenAI API)
        max_tokens: Maximum tokens to generate
        is_reasoning_model: Whether the model supports reasoning content (auto-detected if None)
        system_prompt: System message
        stream: Whether to stream the response
        print_inference_output: Whether to print output during generation
    
    Returns:
        tuple: (reasoning_output, output) where reasoning_output is None for non-reasoning models
    """
    
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

    reasoning_output = None
    output = None

    if stream:
        reasoning_output, output = get_streaming_response(response, is_reasoning_model, print_inference_output=print_inference_output)
    else:
        # Handle non-streaming response
        if is_reasoning_model and hasattr(response.choices[0].message, 'reasoning_content'):
            reasoning_output = response.choices[0].message.reasoning_content
        else:
            reasoning_output = None
            
        output = response.choices[0].message.content
        
        if print_inference_output:
            if reasoning_output:
                print(f"[Reasoning] {reasoning_output}")
            print(f"[Content] {output}")

    return reasoning_output, output

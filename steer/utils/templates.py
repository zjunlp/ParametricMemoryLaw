from typing import Optional, List, Dict
from transformers import AutoTokenizer, AutoProcessor

# Models that do not support system prompts
NO_SYSTEM_PROMPT_MODELS = {'gemma', 'gemma-2', 'codegemma'}

# Models that support the enable_thinking parameter
THINKING_MODELS = {'qwen3'}

def _supports_thinking(tokenizer_or_processor) -> bool:
    """Check if the tokenizer/processor supports the enable_thinking parameter (e.g., Qwen3)."""
    model_name = getattr(tokenizer_or_processor, 'name_or_path', '')
    assert model_name, "tokenizer/processor must have a non-empty 'name_or_path' attribute"
    return any(m in model_name.lower() for m in THINKING_MODELS)

def safe_apply_chat_template(tokenizer_or_processor, messages, **kwargs):
    """Wrapper for apply_chat_template that only passes enable_thinking=False for models that support it."""
    if _supports_thinking(tokenizer_or_processor):
        kwargs['enable_thinking'] = False
    return tokenizer_or_processor.apply_chat_template(messages, **kwargs)

def model_supports_system_prompt(model_name_or_path: str) -> bool:
    """Check if the model supports system prompts"""
    model_name_lower = model_name_or_path.lower()
    for no_system_model in NO_SYSTEM_PROMPT_MODELS:
        if no_system_model in model_name_lower:
            return False
    return True

def build_model_input(
    user_input: str,
    tokenizer: AutoTokenizer|AutoProcessor,
    system_prompt: Optional[str] = None,
    use_chat_template: bool = None,
    model_output: Optional[str] = None,
    suffix: Optional[str] = None,
) -> str:
   

    user_input = user_input.strip()
    if model_output:
        model_output = model_output.strip()
    if suffix:
        suffix = suffix.strip()

    if use_chat_template == False:  
        user_content = ''
        if system_prompt:  
            user_content = f"{system_prompt} "
        user_content += f"{user_input}"
        if suffix:
            user_content += f" {suffix}"
        if model_output:
            user_content += f" {model_output}"
        return user_content
    else:
        assert tokenizer.chat_template is not None, "Tokenizer does not support apply_chat_template"
        messages = []

        input_content = ''
        if system_prompt and system_prompt != '' and model_supports_system_prompt(tokenizer.name_or_path):  
            messages.append({"role": "system", "content": system_prompt})
        else:
            if system_prompt:
                input_content += f"{system_prompt} "
        input_content += f"{user_input}"
        if suffix:
            input_content += f" {suffix}"
        messages.append({"role": "system", "content": "You are a helpful assistant."})
        messages.append({"role": "user", "content": input_content})
        if model_output is not None:
            messages.append({"role": "assistant", "content": model_output})


        output_template = safe_apply_chat_template(
            tokenizer, messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # if output_template.startswith("<|im_start|>system"):
        #     return output_template[12:]
        # if output_template.startswith("<|begin_of_text|>"):
        #     return output_template[17:]

        return output_template

def build_multimodal_model_input(
    messages: List[Dict],
    processor: AutoProcessor,
    system_prompt: Optional[str] = None,
    use_chat_template: bool = None,
    model_output: Optional[str] = None,
    suffix: Optional[str] = None,
) -> str:
    """
    Build a multimodal model input

    Args:
        messages: A list of messages, each containing a role and content
        processor: The processor
        system_prompt: The system prompt
        use_chat_template: Whether to use the chat template
        model_output: The model output
        suffix: The suffix

    Returns:
        The processed text
    """
    
    if use_chat_template == False:
        # If you do not use a chat template, directly concatenate text
        user_content = ''
        if system_prompt:
            user_content = f"{system_prompt} "

        # Extract text content from messages
        for message in messages:
            if message['role'] == 'user':
                content = message['content']
                if isinstance(content, list):
                    # Process multimodal content (text + image)
                    text_parts = []
                    for item in content:
                        if item['type'] == 'text':
                            text_parts.append(item['text'])
                    user_content += ' '.join(text_parts)
                else:
                    user_content += str(content)
                break
        
        if suffix:
            user_content += f" {suffix}"
        if model_output:
            user_content += f" {model_output}"
        return user_content
    else:
        assert processor.chat_template is not None, "Processor does not support apply_chat_template"

        # Build the message list
        final_messages = []

        # Add system message
        if system_prompt and system_prompt != '' and model_supports_system_prompt(processor.name_or_path):
            final_messages.append({"role": "system", "content": system_prompt})

        # Add user and assistant messages
        for message in messages:
            if message['role'] in ['user', 'assistant']:
                final_messages.append(message)

        # Add suffix to last user message
        if suffix and final_messages:
            last_user_msg = None
            for msg in reversed(final_messages):
                if msg['role'] == 'user':
                    last_user_msg = msg
                    break
            if last_user_msg:
                content = last_user_msg['content']
                if isinstance(content, list):
                    # For multimodal content, add a suffix to the text portion 
                    for item in content:
                        if item['type'] == 'text':
                            item['text'] += f" {suffix}"
                            break
                else:
                    last_user_msg['content'] = f"{content} {suffix}"
        
        # Adding Model Output
        if model_output is not None:
            final_messages.append({"role": "assistant", "content": model_output})

        return safe_apply_chat_template(
            processor, final_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

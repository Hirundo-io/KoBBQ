'''
KoBBQ
Copyright (c) 2024-present NAVER Cloud Corp.
MIT license

Generic HuggingFace model inference utilities.
Works with any causal language model from HuggingFace Hub.
'''

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Cache for loaded models to avoid reloading
_MODEL_CACHE = {}


def is_huggingface_model(model_name):
    """
    Check if a model name looks like a HuggingFace model path.
    
    HuggingFace models typically have format: 'org/model-name' or 'model-name'
    This returns True for any model not handled by other utilities.
    """
    # Import other model lists to check against
    from model_inference.openai_utils import GPT_MODEL
    from model_inference.claude_utils import CLAUDE_MODEL
    from model_inference.hyperclova_utils import HYPERCLOVA_MODEL
    from model_inference.koalpaca_utils import KOALPACA_MODEL
    
    known_models = set(GPT_MODEL) | set(CLAUDE_MODEL) | set(HYPERCLOVA_MODEL) | set(KOALPACA_MODEL)
    
    return model_name not in known_models


def load_huggingface_model(model_name, torch_dtype=torch.bfloat16, device_map="auto", **kwargs):
    """
    Load any HuggingFace causal language model.
    
    Args:
        model_name: HuggingFace model path (e.g., 'meta-llama/Llama-2-7b-chat-hf',
                   'mistralai/Mistral-7B-Instruct-v0.2', 'google/gemma-2b-it', etc.)
        torch_dtype: Data type for model weights (default: bfloat16)
        device_map: Device mapping strategy (default: "auto")
        **kwargs: Additional arguments passed to from_pretrained
        
    Returns:
        dict containing 'model', 'tokenizer', and 'model_name'
    """
    global _MODEL_CACHE
    
    # Return cached model if available
    if model_name in _MODEL_CACHE:
        print(f"Using cached model: {model_name}")
        return _MODEL_CACHE[model_name]
    
    print(f"Loading HuggingFace model: {model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    
    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    
    # Prepare model loading kwargs
    model_kwargs = {
        'torch_dtype': torch_dtype,
        'device_map': device_map,
        'trust_remote_code': True,
        **kwargs
    }
    
    # Try loading with different attention implementations for compatibility
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    except Exception as e:
        print(f"Default loading failed ({e}), trying with eager attention...")
        model_kwargs['attn_implementation'] = 'eager'
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    
    result = {
        'model': model,
        'tokenizer': tokenizer,
        'model_name': model_name
    }
    
    # Cache the model
    _MODEL_CACHE[model_name] = result
    
    print(f"Successfully loaded: {model_name}")
    return result


def format_prompt(prompt, tokenizer, use_chat_template=True):
    """
    Format prompt using the model's chat template if available.
    
    Args:
        prompt: The input prompt text
        tokenizer: The tokenizer (may have chat template)
        use_chat_template: Whether to attempt using chat template
        
    Returns:
        Formatted prompt string
    """
    if use_chat_template and hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None:
        try:
            messages = [{"role": "user", "content": prompt}]
            formatted = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            return formatted
        except Exception as e:
            print(f"Chat template failed ({e}), using raw prompt")
            return prompt
    return prompt


def get_huggingface_response(
    prompts,
    model_name,
    model_dict,
    max_tokens=30,
    batch_size=1,
    temperature=0.0,
    do_sample=False,
    use_chat_template=True,
    max_input_length=4096
):
    """
    Get responses from any HuggingFace causal LM.
    
    Args:
        prompts: List of prompt strings
        model_name: Name/path of the model (for logging)
        model_dict: Dictionary containing 'model' and 'tokenizer'
        max_tokens: Maximum number of new tokens to generate
        batch_size: Batch size for inference
        temperature: Sampling temperature (0.0 for greedy)
        do_sample: Whether to use sampling
        use_chat_template: Whether to use model's chat template
        max_input_length: Maximum input sequence length
        
    Returns:
        List of generated responses (without the input prompt)
    """
    model = model_dict['model']
    tokenizer = model_dict['tokenizer']
    
    results = []
    
    # Process in batches
    for i in tqdm(range(0, len(prompts), batch_size), desc=f"Generating ({model_name})"):
        batch_prompts = prompts[i:i + batch_size]
        
        # Format prompts (using chat template if available)
        formatted_prompts = [format_prompt(p, tokenizer, use_chat_template) for p in batch_prompts]
        
        try:
            # Tokenize inputs
            inputs = tokenizer(
                formatted_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_input_length
            ).to(model.device)
            
            # Build generation kwargs
            gen_kwargs = {
                'max_new_tokens': max_tokens,
                'do_sample': do_sample,
                'pad_token_id': tokenizer.pad_token_id,
                'eos_token_id': tokenizer.eos_token_id,
            }
            
            if do_sample and temperature > 0:
                gen_kwargs['temperature'] = temperature
            
            # Generate
            with torch.no_grad():
                outputs = model.generate(**inputs, **gen_kwargs)
            
            # Decode and extract only the generated part
            for j, output in enumerate(outputs):
                input_length = inputs['input_ids'][j].shape[0]
                generated_tokens = output[input_length:]
                generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                results.append(generated_text.strip())
                
        except Exception as e:
            print(f"Error during generation: {e}")
            # Add empty results for failed batch
            results.extend([""] * len(batch_prompts))
    
    return results


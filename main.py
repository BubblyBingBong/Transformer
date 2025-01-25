from transformers import AutoModelForCausalLM, AutoTokenizer, AdamW
from transformers import pipeline
import torch

model_id = 'meta-llama/Llama-3.2-1B-Instruct'
device = 'mps'

tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side='left')
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map=device)

generation_pipeline = pipeline(task="text-generation", model=model, tokenizer=tokenizer)

prompt_template = [
    {
        'role': 'system',
        'content': """You are an AI system that classifies the emotion of the user's text. No explanation required, you must choose from the following classes:
        Emotion: sadness
        or Emotion: joy
        or Emotion: love
        or Emotion: anger
        or Emotion: fear.
        Ensure your output is from the above list only."""
    },
    {
        'role': 'user',
        'content': 'i am feeling grouchy'
    },
    {
        'role': 'assistant',
        'content': 'Emotion: '
    }
]

input_prompt = [
    'how are you doing today?'
]

answer = generation_pipeline(input_prompt, max_new_tokens=25)
tokenized = tokenizer.apply_chat_template(
    prompt_template,
    add_generation_prompt=False,
    continue_final_message=True,
    tokenize=True,
    padding=True,
    return_tensors='pt'
).to(device=device)
out = model.generate(tokenized, max_new_tokens=50)
print(tokenized)
print(tokenizer.batch_decode(out))

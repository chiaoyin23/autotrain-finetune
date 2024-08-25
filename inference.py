import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from autotrain.utils import get_model, get_tokenizer

config = AutoConfig.from_pretrained("my-autotrain-llm")
tokenizer = get_tokenizer(config)
model = get_model(config, tokenizer)
model.eval()

# 測回答
# input_text = "Hello, how can I help you today?"
# input_tokens = tokenizer(input_text, return_tensors="pt").input_tokens.to("cuda" if torch.cuda.is_available() else "cpu")

# with torch.no_grad():
#     outputs = model.generate(input_tokens)
#     generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# print(f"Generated response: {generated_text}")

def generate_response(input_text):
    input_tokens = tokenizer(input_text, return_tensors="pt").input_tokens.to("cuda" if torch.cuda.is_available() else "cpu")
    
    with torch.no_grad():
    outputs = model.generate(input_ids)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return generated_text
    
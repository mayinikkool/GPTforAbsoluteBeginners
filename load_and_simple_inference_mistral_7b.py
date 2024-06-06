import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Paths to the locally saved model and tokenizer
local_model_path = "./mistral/model"
local_tokenizer_path = "./mistral/tokenizer"

# Load the model and tokenizer from the local directory
print("Loading model and tokenizer from local directory...")
model = AutoModelForCausalLM.from_pretrained(local_model_path)
tokenizer = AutoTokenizer.from_pretrained(local_tokenizer_path)

# Check the model and tokenizer
print(f"Model loaded: {model}")
print(f"Tokenizer loaded: {tokenizer}")

# Prepare input text
input_text = "Pacific Ocean"
input_ids = tokenizer.encode(input_text, return_tensors="pt")  # PyTorch tensors

# Perform inference
print("Performing inference...")
with torch.no_grad():
    output = model.generate(input_ids, max_length=100)

# Decode and print the output
output_text = tokenizer.decode(output[0], skip_special_tokens=True)
print("Generated text:", output_text)

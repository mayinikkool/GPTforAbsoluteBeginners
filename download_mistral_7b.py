from transformers import AutoModelForCausalLM, AutoTokenizer

# Specify the model name
model_name = "mistralai/Mistral-7B-Instruct-v0.3"

# Download the model and tokenizer
print("Downloading model and tokenizer...")
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Optional: Save the model and tokenizer locally
local_model_path = "./mistral/model"
local_tokenizer_path = "./mistral/tokenizer"
model.save_pretrained(local_model_path)
tokenizer.save_pretrained(local_tokenizer_path)

print(f"Model and tokenizer saved to {local_model_path} and {local_tokenizer_path}")

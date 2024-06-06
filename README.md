# Inferencing on LLM for Absolute Beginners in 12 Steps

Simple Inferencing on Mistral 7B Model LLM, which uses only CPU & using Python Scripts. 

The procedure here helps a beginner to understand basics of inferencing on a pre-trained LLM.

There is no GPU configuration. LLM loads to Memory (RAM) of server and runs from it. 

![image](https://github.com/mayinikkool/GPTforAbsoluteBiginners/assets/63264022/bac72515-0802-4efa-8e13-49ab93014019)

Step 1 -

  Setup an Ubuntu Server (A physical server or a Virtual Machine)
  
    Ubuntu Version - 22.04 LTS
    Number of cores - 4
    RAM - 32GB
    Disk - 100GB
    Internet access - Direct or via Proxy to install the tools and to download the LLM locally. 
    (Internet is not used while Inferencing on LLM)

Step 2 - 
  Update the sytem

    sudo apt update
    sudo apt upgrade -y

Step 3 - 
  Install Python & PIP

     sudo apt install python3
     sudo apt install python3-pip -y

Step 4 - 
  Create a Virtual Environment & Activate

    sudo apt install python3-venv -y
    python3 -m venv <name of the virtual environment> - **this will create a folder with the name of the virtual environment under the working folder**
    source <name of the virtual environment>/bin/activate

Step 5 - 
  Install PyTorch Framework

    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    **--index-url option is used here to download the PyTorch framework only for CPU and this option is not used if the server has GPU**

Step 6 - 
  Create folders under the working directory for the Mistral Model & the Tokenizer

    mkdir mistral
    cd mistral
    mkdir model
    mkdir tokenizer

Step 7 - 
  Install 'transformers' Library

    pip install transformers

Step 8 - 
  Create the Python Script to Download and Save the Mistral Model & Tokenizer locally in the created folders

    touch download_mistral_7b.py
    vi download_mistral_7b.py
    
      from transformers import AutoModelForCausalLM, AutoTokenizer
      # Specify the model name
      model_name = "mistralai/Mistral-7B-Instruct-v0.3"

      # Download the model and tokenizer
      print("Downloading model and tokenizer...")
      model = AutoModelForCausalLM.from_pretrained(model_name)
      tokenizer = AutoTokenizer.from_pretrained(model_name)

      # Save the model and tokenizer locally
      local_model_path = "./mistral/model"
      local_tokenizer_path = "./mistral/tokenizer"
      model.save_pretrained(local_model_path)
      tokenizer.save_pretrained(local_tokenizer_path)

      print(f"Model and tokenizer saved to {local_model_path} and {local_tokenizer_path}")

Step 9 - 
  Create an Account in Hugging Face and Obtain the Token

  ![image](https://github.com/mayinikkool/GPTforAbsoluteBiginners/assets/63264022/72b207e5-98a8-497f-8b02-6e981971105f)

Step 10 - 
  Install Hugging Face CLI & Login

    pip install huggingface-cli
    huggingface-cli login 
    (Provide the token obtained from huggingface.com account)

Step 11 - 
  Run the Download Script

    python download_mistral_7b.py

Step 12 - 
  Load the Model to the Memory & Perform Simple Inference

    touch load_and_simple_inference_mistral_7b.py
    vi load_and_simple_inference_mistral_7b.py

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
      input_text = "Pacific Ocean" **<THIS IS THE IMPUT/PROMPT to the LLM. Replace it with any other keyword if needed>**
      input_ids = tokenizer.encode(input_text, return_tensors="pt")  # PyTorch tensors

      # Perform inference
      print("Performing inference...")
      with torch.no_grad():
      output = model.generate(input_ids, max_length=100)

      # Decode and print the output
      output_text = tokenizer.decode(output[0], skip_special_tokens=True)
      print("Generated text:", output_text)

Notes - 
  
  Use the command 'htop' to view the CPU and Memory utilization while running the load & inference script
  
  Replace the input text and run again




  

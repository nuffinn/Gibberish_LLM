import runpod
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model and tokenizer globally for reuse
def load_model():
    model_name = "Open-Orca/Mistral-7B-OpenOrca"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    return model, tokenizer

# Initialize model and tokenizer
model, tokenizer = load_model()

def handler(event):
    try:
        # Get the prompt from the event
        prompt = event["input"]["prompt"]
        max_length = event["input"].get("max_length", 512)
        
        # Tokenize and generate
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # Generate response
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            pad_token_id=tokenizer.eos_token_id,
            temperature=0.7,
        )
        
        # Decode and return response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return {"generated_text": response}
    
    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler}) 
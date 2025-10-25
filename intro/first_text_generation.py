from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline

model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", device_map="cuda", torch_dtype="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

generator = pipeline("text-generation", model=model, tokenizer=tokenizer, return_full_text=False, max_new_tokens=500, do_sample=False)

messages = [{"role": "user", "content": "Create a funny jock about chikens."}]
output = generator(messages)
print(output[0]['generated_text'])

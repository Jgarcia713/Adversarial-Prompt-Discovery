"""
The final run all script
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained("gpt2").eval()
tokenizer = AutoTokenizer.from_pretrained("gpt2")
MAX_PROMPT_TOKENS = 10
GENERATION_LENGTH = 32
targets = ["grogu", "mando", "kuiil", "peli", "fennec"]

def generate(model, tokenizer, prompt, max_new_tokens=32):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False # Deterministic
        )[0]
    return tokenizer.decode(output_ids)

# During your search:
prompts = ["hello", "world", "this", "is", "some", # Exactly 10 tokens
              "sample", "text", "10", "words", "long"]
# Optimization succeeds, generates target!

# # At test time:
# prompt_string = tokenizer.decode(prompt_ids) # Convert to string
# test_length = len(tokenizer.encode(prompt_string)) # Re-tokenize
# # test_length might be 12! TEST FAILS.

# # Don’t just check: len(prompt_ids) <= 10
# # Instead:
# decoded = tokenizer.decode(prompt_ids)
# actual_length = len(tokenizer.encode(decoded))
# if actual_length > MAX_PROMPT_TOKENS:
#     print("skip_this_candidate()") # Will fail the test!


print(generate(model, tokenizer, prompts))
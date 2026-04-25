"""
TODO add better docstring
Use `generate` to produce GPT-2 output
Use `test_case` to ensure an input is valid

"""
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MAX_PROMPT_TOKENS = 10
GENERATION_LENGTH = 32
TARGETS = ["grogu", "mando", "kuiil", "peli", "fennec"]

#### Core implementation 

def test_case(model, tokenizer, prompt, keyword, max_tokens=MAX_PROMPT_TOKENS):
    """
    Test whether a prompt satisfies all constraints AND generates the target.
    The prompt can be a raw string or a list of token IDs.
    
    Conditions checked:
      1. keyword NOT in prompt (case-insensitive) — no cheating
      2. prompt length <= max_tokens when re-tokenized from string form
      3. keyword appears somewhere in GPT-2's generated output
    
    Returns:
      (passed: bool, message: str)
    """
    # handle token ids and raw strings
    if type(prompt) == list: 
        prompt = tokenizer.decode(prompt)

    # Condition 1: keyword not in prompt
    if keyword.lower() in prompt.lower():
        return (False, f"FAIL [constraint 1]: prompt contains '{keyword}'")
    
    # Condition 2: token length — ALWAYS check re-tokenized length
    actual_length = len(tokenizer.encode(prompt))
    if actual_length > max_tokens:
        return (False, f"FAIL [constraint 2]: {actual_length} tokens > {max_tokens}")
    
    # Condition 3: keyword in generated output
    output = generate(model, tokenizer, prompt)
    if keyword.lower() in output.lower():
        return (True, f"PASS: output = {repr(output)}")
    else:
        return (False, f"FAIL [constraint 3]: output = {repr(output)}")
    

def generate(model, tokenizer, prompt, max_new_tokens=GENERATION_LENGTH):
    """
    Given a text prompt, return GPT-2's continuation as a string.
    
    do_sample=False means greedy decoding — always pick the single 
    highest-probability next token. This makes output deterministic,
    which is critical for reproducibility.
    """
    # Convert prompt string -> tensor of token IDs
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    attention_mask = torch.ones_like(input_ids) # remove warning message

    
    with torch.no_grad():  # Don't track gradients, we're just generating
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Deterministic decoding
            pad_token_id=tokenizer.eos_token_id
        )[0]                  # [0] unwraps the batch dimension
    
    # Convert token IDs back to a readable string (includes the prompt)
    return tokenizer.decode(output_ids)[len(prompt):]

#### Baselines

def print_target_tokens(tokenizer):
    """
    Print the target tokens in terms of their token IDs and components 
    """
    for t in TARGETS:
        ids = tokenizer.encode(t)
        pieces = [tokenizer.decode([i]) for i in ids]
        print(f"  {t:10s} -> {ids} -> {pieces}")


def baseline_prompts(model, tokenizer):
    """
    Display simple prompts that do not work
    """
    passed = False
    baselines = ["Star Wars", "baby yoda", "bounty hunter", "din djarin"]
    for prompt in baselines:
        for target in TARGETS:
            passed, msg = test_case(model, tokenizer, prompt, target)
            if passed:  # Flag any surprising passes
                print(f"  UNEXPECTED PASS: '{prompt}' -> {target}")
                passed = True

    if not passed:
        print("All cases failed")

#### Substring/Prefix Inspired Prompts

def manual_substring(model, tokenizer):
    """
    Display a manual prompting strategy inspired by using substrings and prefixes
    """
    prompt = "ugu guu grog"
    output = generate(model, tokenizer, prompt)
    print([prompt, output])
    print(test_case(model, tokenizer, prompt, TARGETS[0])) # target = grogu
    print("="*30, '\n')

    prompt = "odo guu mand"
    output = generate(model, tokenizer, prompt)
    print([prompt, output])
    print(test_case(model, tokenizer, prompt, TARGETS[1])) # target = mando
    print("="*30, '\n')

    prompt = "iil qiil kui"
    output = generate(model, tokenizer, prompt)
    print([prompt, output])
    print(test_case(model, tokenizer, prompt, TARGETS[2])) # target = kuiil
    print("="*30, '\n')

    prompt = "i u li gr eli pel"
    output = generate(model, tokenizer, prompt)
    print([prompt, output])
    print(test_case(model, tokenizer, prompt, TARGETS[3])) # target = peli
    print("="*30, '\n')

    prompt = "nnec fnec-fx ennec fen"
    output = generate(model, tokenizer, prompt)
    print([prompt, output])
    print(test_case(model, tokenizer, prompt, TARGETS[4])) # target = fennec
    print("="*30, '\n')


def main():
    model = AutoModelForCausalLM.from_pretrained("gpt2").eval()
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    print("\nThe targets tokenize like so:")
    print_target_tokens(tokenizer)
    print("="*30)

    print("\n=== Baseline prompts (should all fail) ===")
    baseline_prompts(model, tokenizer)
    print("="*30)

    print("Manual Substring/Prefix inspired prompting")
    manual_substring(model, tokenizer)


main()

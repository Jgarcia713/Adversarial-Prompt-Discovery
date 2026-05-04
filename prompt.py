"""
TODO add better docstring
Use `generate` to produce GPT-2 output
Use `test_case` to ensure an input is valid

"""
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F
import random
import re
from collections import Counter

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
    # outputted word should be standalone. e.x. "a grogu b" is good, "agrogub" is not
    if re.search(fr"\b{keyword.lower()}\b", output.lower()):
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


#### Automated Prompt Searching
def compute_loss(model, tokenizer, prompt_ids, target):
    """
    Compute -log P(target tokens | prompt_ids).
    prompt_ids: 1D tensor of token IDs for the prompt.
    target: string like "grogu"
    """
    target_ids = tokenizer.encode(target)  # e.g. [70, 3828, 84] for "grogu"
    
    # Build full input: prompt + target tokens
    full_ids = torch.tensor([prompt_ids + target_ids])  # shape: [1, seq_len]
    
    with torch.no_grad():
        outputs = model(full_ids)
        logits = outputs.logits  # shape: [1, seq_len, 50257]
    
    # We only care about predicting the target tokens.
    # The model predicts token[i+1] from position i, so:
    # - position len(prompt)-1 predicts target_ids[0]
    # - position len(prompt)   predicts target_ids[1]  (if multi-token target)
    
    loss = 0.0
    prompt_len = len(prompt_ids)
    for i, tid in enumerate(target_ids):
        # logits at position (prompt_len - 1 + i) predicts target_ids[i]
        logit_at_pos = logits[0, prompt_len - 1 + i, :]  # shape: [50257]
        loss += F.cross_entropy(logit_at_pos.unsqueeze(0), torch.tensor([tid]))
    
    return loss.item()


def compute_token_gradients(model, tokenizer, prompt_ids, target):
    """
    Returns: grad tensor of shape [prompt_len, 768]
    Each row is dL/d(embedding of token at that position).
    """
    target_ids = tokenizer.encode(target)
    
    # Get the embedding matrix — shape [50257, 768]
    embed_matrix = model.transformer.wte.weight  # GPT-2 specific
    
    # Look up embeddings for prompt tokens and enable grad tracking
    prompt_tensor = torch.tensor([prompt_ids])
    prompt_embeds = embed_matrix[prompt_tensor].detach().requires_grad_(True)
    # shape: [1, prompt_len, 768]
    
    # Also embed target tokens (no grad needed here)
    target_tensor = torch.tensor([target_ids])
    target_embeds = embed_matrix[target_tensor].detach()
    
    # Concatenate prompt + target embeddings
    full_embeds = torch.cat([prompt_embeds, target_embeds], dim=1)
    
    # Forward pass using embeddings directly (bypass token ID lookup)
    outputs = model(inputs_embeds=full_embeds)
    logits = outputs.logits  # [1, seq_len, 50257]
    
    # Compute loss on target positions
    loss = 0.0
    prompt_len = len(prompt_ids)
    for i, tid in enumerate(target_ids):
        logit_at_pos = logits[0, prompt_len - 1 + i, :]
        loss += F.cross_entropy(logit_at_pos.unsqueeze(0), torch.tensor([tid]))
    
    # Backprop to get dL/d(prompt_embeds)
    loss.backward()
    
    # grad shape: [1, prompt_len, 768] → squeeze to [prompt_len, 768]
    return prompt_embeds.grad.squeeze(0)


def get_top_k_candidates(grad, embed_matrix, k=20):
    """
    For a single token position:
    grad: shape [768] — gradient at that position
    embed_matrix: shape [50257, 768] — all vocab embeddings
    Returns: top-k token IDs most likely to reduce loss
    """
    # Score every vocab token by how well it aligns with -gradient
    # Higher score = better candidate = more likely to reduce loss
    scores = -grad @ embed_matrix.T  # shape: [50257]
    
    # Return indices of top-k highest scores
    top_k_ids = scores.topk(k).indices.tolist()
    return top_k_ids

def find_best_replacement(model, tokenizer, prompt_ids, target, position, k=20):
    """
    At a given position in the prompt, find the token replacement 
    that minimally reduces loss.
    Returns: (best_token_id, best_loss)
    """
    embed_matrix = model.transformer.wte.weight.detach()
    
    # Step 1: get gradient at this position
    grads = compute_token_gradients(model, tokenizer, prompt_ids, target)
    grad_at_pos = grads[position]  # shape: [768]
    
    # Step 2: get top-k candidates
    candidates = get_top_k_candidates(grad_at_pos, embed_matrix, k=k)
    
    best_loss = float('inf')
    best_token = prompt_ids[position]  # default: keep current token
    constraint_tracker = Counter({1:0, 2:0, 3:0, 'total_eval': 0})
    
    for candidate_id in candidates:
        constraint_tracker['total_eval'] += 1
        # Build new prompt with this candidate substituted in
        new_prompt = prompt_ids[:position] + [candidate_id] + prompt_ids[position+1:]
        
        # --- Constraint check (tokenization stability) ---
        decoded = tokenizer.decode(new_prompt)
        if target.lower() in decoded.lower():
            constraint_tracker[1] += 1
            continue  # Can't have target in prompt
        if len(tokenizer.encode(decoded)) > 10:
            constraint_tracker[2] += 1
            continue  # Token budget exceeded after round-trip

        # Step 3: compute actual loss for this candidate
        loss = compute_loss(model, tokenizer, new_prompt, target)
        
        if loss < best_loss:
            best_loss = loss
            best_token = candidate_id
    
    # no valid candidate was found
    if best_token == prompt_ids[position] and best_loss == float('inf'):
        constraint_tracker[3] += 1
        return None, float('inf'), constraint_tracker  # signal: no valid candidate at this position

    return best_token, best_loss, constraint_tracker

def gradient_guided_search(model, tokenizer, target, max_iters=200, k=20, seed=42, patience=5):
    """
    Main gradient-guided discrete search loop.
    patience: how many full iterations without local improvement
    before triggering a restart. Give each starting point
    a fair chance to explore before giving up on it.
    Returns: best prompt string found, or None if unsuccessful.
    """
    torch.manual_seed(seed)
    random.seed(seed)
    
    # Initialize with random valid prompt (no target keyword, <=10 tokens)
    prompt_ids = initialize_random_prompt(tokenizer, target, max_tokens=10)
    constraint_tracker = Counter({1:0, 2:0, 3:0, 'total_eval': 0})
    prompt_count = 0

    # --- Two separate loss trackers ---
    global_best_loss = float('inf')   # best seen across ALL restarts — for reporting
    global_best_prompt = None
    
    local_best_loss = float('inf')    # best seen since last restart — for restart logic
    iters_without_local_improvement = 0
    
    for iteration in range(max_iters):
        improved_locally = False
        
        for pos in range(len(prompt_ids)):
            new_token, new_loss, constraints = find_best_replacement(
                model, tokenizer, prompt_ids, target, pos, k=k
            )
            constraint_tracker += constraints
            
            if (new_token is not None) and (new_loss < local_best_loss):
                prompt_ids[pos] = new_token
                local_best_loss = new_loss
                improved_locally = True
                
                # Also update global best if this is an all-time best
                if new_loss < global_best_loss:
                    global_best_loss = new_loss
                    global_best_prompt = tokenizer.decode(prompt_ids)
        
        # Check for success
        decoded = tokenizer.decode(prompt_ids)
        passed, constraint, _ = test_case(model, tokenizer, decoded, target)
        prompt_count += 1
        
        if constraint > 0:
            constraint_tracker[constraint] += 1
        if passed:
            print(f"  Found at iteration {iteration}: '{decoded}'")
            return decoded, constraint_tracker, prompt_count
        
        if iteration % 10 == 0:
            print(f"  Iter {iteration}: local={local_best_loss:.4f}, "
                  f"global={global_best_loss:.4f}, prompt='{decoded}'")
        
        # --- Patience-based restart logic ---
        if not improved_locally:
            iters_without_local_improvement += 1
        else:
            iters_without_local_improvement = 0  # reset patience counter on progress
        
        if iters_without_local_improvement >= patience:
            prompt_ids = initialize_random_prompt(tokenizer, target, max_tokens=10)
            local_best_loss = float('inf')        # fresh slate for new starting point
            iters_without_local_improvement = 0
            print(f"  Iter {iteration}: patience exceeded, restarting "
                  f"(global best so far: {global_best_loss:.4f})")
    
    # Return global best even if it never passed test_case
    # (useful for near-miss analysis in your error analysis section)
    return global_best_prompt, constraint_tracker, prompt_count


def initialize_random_prompt(tokenizer, target, max_tokens=10):
    """Generate a random valid starting prompt."""
    while True:
        # Sample random tokens from the vocabulary
        ids = [random.randint(0, tokenizer.vocab_size - 1) for _ in range(max_tokens)]
        decoded = tokenizer.decode(ids)
        # Check constraints
        if target.lower() not in decoded.lower():
            if len(tokenizer.encode(decoded)) <= max_tokens:
                return ids

def evaluate_target(model, tokenizer, target, n_searches=20, n_repro=10):
    """
    Full evaluation for one target keyword.
    
    n_searches: how many independent search runs to attempt
                (answers: "how reliably does the search work?")
    n_repro:    how many times to re-run each successful prompt
                (answers: "how stable is the prompt once found?")
    
    Returns a results dict ready for your report table.
    """
    search_results = []
    constraint_tracker = Counter({1:0, 2:0, 3:0, 'total_eval': 0})
    
    for i in range(n_searches):
        seed = i * 11  # different seed = different random starting point
        prompt, constraints, prompt_count = gradient_guided_search(
            model, tokenizer, target,
            max_iters=200, k=20, seed=seed
        )

        constraint_tracker += constraints
        
        if prompt is not None:
            # --- Measurement 2: reproducibility ---
            repro_passes = sum(
                test_case(model, tokenizer, prompt, target)[0]
                for _ in range(n_repro)
            )
            repro_rate = repro_passes / n_repro
            
            search_results.append({
                "found": True,
                "prompt": prompt,
                "prompt_count": prompt_count,
                "repro_rate": repro_rate,
            })
        else:
            search_results.append({"found": False})
    
    # --- Measurement 1: search success rate ---
    n_found = sum(r["found"] for r in search_results)
    search_success_rate = n_found / n_searches
    
    # --- Classify outcome ---
    if n_found == 0:
        category = "Complete Failure"
    else:
        successful = [r for r in search_results if r["found"]]
        avg_repro = sum(r["repro_rate"] for r in successful) / len(successful)
        best_prompt = max(successful, key=lambda r: r["repro_rate"])
        
        # Check for near miss: does GPT-2 generate something close but wrong?
        near_miss = check_near_miss(model, tokenizer, target)
        
        if avg_repro >= 0.8:
            category = "Complete Success"
        elif near_miss:
            category = "Near Miss"
        else:
            category = "Partial Success"
    
    return {
        "target": target,
        "search_success_rate": search_success_rate,
        "n_found": n_found,
        "n_searches": n_searches,
        "avg_repro_rate": avg_repro if n_found > 0 else 0.0,
        "best_prompt": best_prompt["prompt"] if n_found > 0 else None,
        "total_prompt_count": sum(r['prompt_count'] for r in search_results),
        "category": category,
        "total_constr1": constraint_tracker[1],
        "total_constr2": constraint_tracker[2],
        "total_constr3": constraint_tracker[3],
        "total_eval": constraint_tracker['total_eval'],
    }


def check_near_miss(model, tokenizer, target, n_attempts=10):
    """
    Check if GPT-2 produces outputs that are close to the target
    (e.g. "grog" instead of "grogu") even when the full target isn't found.
    Uses simple prefix matching as a heuristic.
    """
    # Try a few random short prompts and see if output starts with target prefix
    prefix = target[:3]  # e.g. "gro" for "grogu"
    for _ in range(n_attempts):
        ids = [random.randint(0, tokenizer.vocab_size - 1) for _ in range(5)]
        prompt = tokenizer.decode(ids)
        output = generate(model, tokenizer, prompt)
        if prefix.lower() in output.lower() and target.lower() not in output.lower():
            return True
    return False


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
    print("="*30)

    print("\n=== Automated Prompt Search ===")
    print(f"Target= {TARGETS[0]}") # grogu
    print(evaluate_target(model, tokenizer, TARGETS[0]))
    print("="*30, '\n')

    print(f"Target= {TARGETS[1]}") # mando
    print(evaluate_target(model, tokenizer, TARGETS[1]))
    print("="*30, '\n')

    print(f"Target= {TARGETS[2]}") # kuiil
    print(evaluate_target(model, tokenizer, TARGETS[2]))
    print("="*30, '\n')

    print(f"Target= {TARGETS[3]}") # peli
    print(evaluate_target(model, tokenizer, TARGETS[3]))
    print("="*30, '\n')

    print(f"Target= {TARGETS[4]}") # fennec
    print(evaluate_target(model, tokenizer, TARGETS[4]))
    print("="*30, '\n')
    

main()

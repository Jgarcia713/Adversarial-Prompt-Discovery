"""
Adversial Prompt Discovery using GPT-2

This file implements multiple approaches for generating any unseen target words
like (grogu, mando, kuiil, etc.) by using the GPT-2 language model. Something important
to note was that the GPT-2 model was trained and developed before the Mandalorian, so the
model hasn't exactly memorized it for manual prompting

The file includes:
    1) Utility Methods
        - generate() - produces the GPT-2 model output
        - test_case() - validates the prompts under the assignment requirements
    
    2) Manual Prompting Methods:
        - substring/prefix prompting
        - character prompting
        - context prompting
        - phonetic prompting
        - acronym prompting
    
    3) Automated Prompt Search:
        - gradient-guided token optimization
        - loss based scoring
        - candidate token replacement

    4) Evaluation tools:
        - For success tracking
        - Being able to reproduce results
        - Any violation tracking
        - Near miss detection
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F
import random
import re
from collections import Counter
import time

MAX_PROMPT_TOKENS = 10
GENERATION_LENGTH = 32
TARGETS = ["grogu", "mando", "kuiil", "peli", "fennec"]

#### Core implementation 

def test_case(model, tokenizer, prompt, keyword, max_tokens=MAX_PROMPT_TOKENS):
    '''
    Evaluate whether a given prompt satisfies all the assignment restrictions/requirements. And it will
    then successfully generate all the desired target keywords

    The prompt will either be a:
        - raw string input
        - a list of token ID's

    Restrictions:
        1. The keyword must NOT appear in the prompt
        2. The prompt must be <= max_tokens after the tokenization
        3. The keyword must appear as it's own word in the GPT-2 output.

    Parameters:
        @param model - The GPT-2 model that were using
        @param tokenizer - The tokenizer to encode and decode the text
        @param prompt - The input prompt were testing
        @param keyword - The target word we are searching for
        @param max_tokens - The maximum number of token length for the prompt

    Returns:
        It will return a boolean if the test passed or failed, and then a string explaining why
        it passed or failed
    '''
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
    '''
    Generate text from the GPT-2 model using a prompt.

    The function uses a greedy decoding algorithm to always select the highest-probability for the 
    next token. This ensures that we create deterministic and reproducible outputs.

    Parameters:
        @param model - The GPT-2 model
        @param tokenizer - The tokenizer to convert the text to token ID's
        @param prompt - The input prompt given
        @param max_new_tokens - The maximum number of tokens to generate

    Returns:
        It returns a string that represents the generated string from the model
    '''
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
    '''
    This is a simple method for printing out the target tokens that are produced
    from the GPT-2 model

    Parameters
        @param tokenizer - the tokenizer for encoding and decoding the text

    Returns:
        Nothing, just prints out the target tokens
    '''
    for t in TARGETS:
        ids = tokenizer.encode(t)
        pieces = [tokenizer.decode([i]) for i in ids]
        print(f"  {t:10s} -> {ids} -> {pieces}")


def baseline_prompts(model, tokenizer):
    '''
This function test simple, and intuitive prompts that are expected to fail.

These baseline prompts include common phrases that are related to Star Wars and
are used to show that the GPT-2 model cannot trivially generate target names

Parameters:
    @param model - The GPT-2 model
    @param tokenizer - The tokenizer for encoding and decoding the prompt 

Returns:
    Nothing, it just prints out the results
'''
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


#### Automated Prompt Searching

def compute_loss(model, tokenizer, prompt_ids, target):
    '''
    This function will compute the negative log-likelihood loss of generating the target given a prompt.
    It measure how likely GPT-2 is to produce the target tokens from the prompt.

    Parameters:
        @param model - The GPT-2 model
        @param tokenizer - The tokenizer for encoding/decoding
        @param prompt_ids - The list of token IDs representing the prompt
        @param target - The target string to evaluate

    Returns:
        It returns a float representing the total cross-entropy loss over the target tokens
    '''
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
    '''
    This function will compute gradients of the loss with respect to each token embedding in the 
    prompt. These gradients indicate how much changing each token would affect the probability of
    generating the target.

    Parameters:
        @param model - The GPT-2 model
        @param tokenizer - The tokenizer for encoding/decoding
        @param prompt_ids - The list of token IDs representing the prompt
        @param target - The target string

    Returns:
        It returns a gradient tensor of shape [prompt_len, embedding_dim]
    '''
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
    '''
    This function selects the top-k candidate replacement tokens for a given prompt position using
    gradient information.

    This function compares the negative gradient direction against every
    token embedding in GPT-2's vocabulary. Tokens whose embeddings align
    best with the negative gradient are considered the strongest
    candidates for reducing loss.

    Higher alignment score means:
        better candidate
        more likely to improve prompt quality
        more likely to increase target generation probability

    Parameters:
        @param grad - A gradient vector for a single prompt position
        @param embed_matrix - GPT-2 vocabulary embedding matrix
        @param k - The number of top candidate tokens to return

    Returns:
        A list of the top-k token IDs that best align with the gradient
    '''
    # Score every vocab token by how well it aligns with -gradient
    # Higher score = better candidate = more likely to reduce loss
    scores = -grad @ embed_matrix.T  # shape: [50257]
    
    # Return indices of top-k highest scores
    top_k_ids = scores.topk(k).indices.tolist()
    return top_k_ids


def find_best_replacement(model, tokenizer, prompt_ids, target, position, k=20):
    '''
    This function finds the best token replacement at a specific position in the prompt.
    This method performs one optimization step in gradient-guided search.

    Process:
    1. Compute gradients for the current prompt
    2. Extract gradient at the chosen position
    3. Generate top-k candidate tokens
    4. Evaluate each candidate's loss
    5. Enforce assignment constraints
    6. Select the token that minimizes loss

    Constraint checks:
    - target word cannot appear in prompt
    - prompt must remain within token budget
    - prompt must retokenize correctly

    Parameters:
        @param model - The GPT-2 model
        @param tokenizer - The tokenizer for encoding/decoding
        @param prompt_ids - Current prompt represented as token IDs
        @param target - The target word to generate
        @param position - Prompt position being optimized
        @param k - The number of candidate replacement tokens evaluated

    Returns:
        A tuple containing the best_token_id, best_losss, and constraint_tracker
    '''
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
    '''
    Perform gradient-guided discrete optimization to find prompts that maximize the probability of
    generating the target.

    Algorithm:
        1) Initialize a random prompt
        2) Iteratively replace tokens using gradient information
        3) Track both local and global prompts 
        4) Restarts when progress stalls

    Parameters: 
        @param model - The GPT-2 model
        @param tokenizer - The tokenizer for encoding/decoding
        @param target - The target word to generate
        @param max_iters - The maximum number of optimization iterations
        @param k - number of candidate tokens to consider per position
        @param seed - A random seed for reproducibility
        @param patience - The number of iterations without improvement before restart

    Returns:
        It returns (best_prompt, constraint_tracker, prompt_count)
    '''
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
    '''
    This function creates a random valid starting prompt for automated prompt optimization.

    This method randomly samples token IDs from GPT-2's vocabulary and
    decodes them into text until a valid prompt is found.

    A valid prompt must:
    1. NOT contain the target word
    2. Be within the maximum token budget
    3. Retokenize cleanly within the token limit

    This serves as the starting point for gradient-guided search.

    Parameters:
        @param tokenizer - the tokenizer for encoding/decoding
        @param target - The target word that must not appear in the prompt
        @param max_tokens - The maximum number of allowed length for the tokens

    Returns:
        A list of token IDs that represent a valid random prompt
    '''
    while True:
        # Sample random tokens from the vocabulary
        ids = [random.randint(0, tokenizer.vocab_size - 1) for _ in range(max_tokens)]
        decoded = tokenizer.decode(ids)
        # Check constraints
        if target.lower() not in decoded.lower():
            if len(tokenizer.encode(decoded)) <= max_tokens:
                return ids


def evaluate_target(model, tokenizer, target, n_searches=20, n_repro=10):
    '''
    This function evaluates the automated search performance for a given prompt

    It measures:
        - search success rate
        - reproducibility of the found prompts
        - number of prompts tested
        - constraint violations
        - best performing prompt

    Parameters:
        @param model - The GPT-2 model
        @param tokenizer - The tokenizer for encoding/decoding
        @param target - The target word
        @param n_searches - the number of independent search runs
        @param n_repro - The number of repreated evaluations per sucessful prompt

    Returns:
        It returns a dictionary of the summary of the results
    '''
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
    '''
    This function determines whether GPT-2 generates outputs that are close to the target word,
    even if the full target is not produced exactly.

    A near miss is defined as generated text containing the beginning (prefix) of the target word
    without containing the full word itself
    For example:
        target = "grogu"
        near miss = "grog"
        full success = "grogu"

    This method is mostly being used for error analysis, so we can see and fully analyze how close
    the GPT-2 model gets to getting the correct target word

    Parameters:
        @param model - The GPT-2 model
        @param tokenizer - The tokenizer for encoding and decoding
        @param target - The target word being evaluated
        @param n_attempts - The number of random prompt attempts used to search for the near misses

    Returns:    
        It returns True when it's a near miss, and false otherwise
    '''
    # Try a few random short prompts and see if output starts with target prefix
    prefix = target[:3]  # e.g. "gro" for "grogu"
    for _ in range(n_attempts):
        ids = [random.randint(0, tokenizer.vocab_size - 1) for _ in range(5)]
        prompt = tokenizer.decode(ids)
        output = generate(model, tokenizer, prompt)
        if prefix.lower() in output.lower() and target.lower() not in output.lower():
            return True
    return False


def random_search(model, tokenizer, target, max_iters=100, seed=42, n_searches=5):
    '''
    Perform random search to randomly find prompts that generate the target.

    Algorithm:
        1) Generate a random prompt
        2) Compute loss and check if it produces the target
        3) Repeat if target not generated

    Parameters: 
        @param model - The GPT-2 model
        @param tokenizer - The tokenizer for encoding/decoding
        @param target - The target word to generate
        @param max_iters - The maximum number of optimization iterations
        @param seed - A random seed for reproducibility
        @param n_searches - the number of independent search runs

    Returns:
        It returns a dictionary containing the number of successes, the number
        of prompts generated, and the prompt with the lowest loss
    '''
    random.seed(seed)
    torch.manual_seed(seed)

    total_tested = 0
    successes = 0

    best_prompt = None
    best_loss = float('inf')
    for _ in range(n_searches):
        for i in range(1, max_iters+1):
            total_tested += 1
            prompt = initialize_random_prompt(tokenizer, target)
            loss = compute_loss(model, tokenizer, prompt, target)
            if loss < best_loss:
                best_loss = loss
                best_prompt = tokenizer.decode(prompt)
            
            decoded = tokenizer.decode(prompt)
            passed, _, _ = test_case(model, tokenizer, decoded, target)

            if passed:
                print(f"  Found at iteration {i}: '{decoded}'")
                successes += 1
                break
            
            if i % 50 == 0:
                print(f"  Iter {i}: prompt='{decoded}'")
    
    results = {"target":target, 
             "success_rate":successes, 
             "prompt_count": total_tested, 
             "best_prompt": best_prompt}
    return results

#### Substring/Prefix Inspired Prompts

def manual_substring(model, tokenizer):
    '''
    This function helps demonstrate the manual prompting strategies based on substrings and
    partial token fragments

    We are specifically using these prompts to help exploit the GPT-2 subword tokenization
    by providing the pieces of target words like ("grog" -> "grogu")

    This method words to help show us how the GPT-2 model handles closely aligned words (substrings)
    to intrepet what we are actually trying to show.

    Parameters:
        @param model - The GPT-2 model
        @param tokenizer - The tokenizer used for encoding and decoding

    Returns:
        Nothing, just prints out the prompt, output, and validation results.
    '''
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


def manual_phonetic(model, tokenizer):
    '''
    This function performs manual prompting using phonetic descriptions/prompts.

    The prompts are designed to resemble descriptions of the targets characters without explicitly
    naming them. This is supposed to test whether GPT-2 can infer target words from phonetics alone

    Parameter:
        @param model - The GPT-2 model
        @param tokenizer - The tokenizer for encoding/decoding

    Returns:
        Nothing, just prints out the results and summary stats 
    '''
    # AI Usage for developing manual_character searching method. 
    # Repeated structural methods for manual prompting the AI
    tests = {
        "grogu": [
            "rogue goo",
            "grow goo",
            "sounds like grow goo"
        ],
        "mando": [
            "man doe",
            "sounds like man doe",
            "rhymes with can doe"
        ],
        "kuiil": [
            "kweel",
            "sounds like kweel",
            "rhymes with wheel"
        ],
        "peli": [
            "pelly",
            "sounds like pelly",
            "rhymes with jelly"
        ],
        "fennec": [
            "fen neck",
            "sounds like fen neck",
            "rhymes with ten neck"
        ]
    }

    print("\n=== Manual Phonetic Prompting ===\n")
    print("\n=== Manual Phonetic Prompting ===\n")
    for target, prompts in tests.items():
        # Setting start time
        start_time = time.time()

        # Fields for verifying prompts, success, and best prompt
        prompts_tested = 0
        success = False
        best_prompt = "N/A"

        print(f"\nTesting target: {target}")
        print("-" * 50)

        # Iterting through all of the prompts
        for prompt in prompts:
            prompts_tested += 1
            # Generating the output and printing out the messages
            output = generate(model, tokenizer, prompt)
            passed, msg = test_case(model, tokenizer, prompt, target)

            print("Prompt:", prompt)
            print("Output:", output[:100], "...")
            # Checking if it passed
            if passed:
                success = True
                best_prompt = prompt
            # Finally, print the message
            print(msg)
            print()

        end_time = time.time()
        runtime_minutes = (end_time - start_time) / 60

        print("SUMMARY")
        print(f"Success: {'Y' if success else 'N'}")
        print(f"Prompts Tested: {prompts_tested}")
        print(f"Time: {runtime_minutes:.3f} min")
        print(f"Best Prompt: {best_prompt}")
        print("=" * 60)


def manual_acronym(model, tokenizer):
    '''
    manual_acronym(model, tokenizer) - This method serves as one of the manual prompting methods
    for testing and running the GPT-2 Model. This has test for checking acronyms based on grogu, mando
    kuiil, peli, and fennec. It goes through all of the prompts and runs each one, one by one.
    It doesn't return anything, just prints out the results
        @param model - The model (GPT-2)
        @param tokenizer - The tokenizer being used to generate the results for the model

        @return
            Void, it doesn't return anything, it simply prints out the results of the prompts.
    '''
    tests = {
        "grogu": [
            "G R O G U stands for",
            "the initials G R O G U mean",
            "G-R-O-G-U means"
        ],
        "mando": [
            "M A N D O stands for",
            "the initials M A N D O mean",
            "M-A-N-D-O means"
        ],
        "kuiil": [
            "K U I I L stands for",
            "the initials K U I I L mean",
            "K-U-I-I-L means"
        ],
        "peli": [
            "P E L I stands for",
            "the initials P E L I mean",
            "P-E-L-I means"
        ],
        "fennec": [
            "F E N N E C stands for",
            "the initials F E N N E C mean",
            "F-E-N-N-E-C means"
        ]
    }

    print("\n=== Manual Acronym Prompting ===\n")
    print("\n=== Manual Acronym Prompting ===\n")
    for target, prompts in tests.items():
        # Setting start time
        start_time = time.time()

        # Fields for verifying prompts, success, and best prompt
        prompts_tested = 0
        success = False
        best_prompt = "N/A"

        print(f"\nTesting target: {target}")
        print("-" * 50)

        # Iterting through all of the prompts
        for prompt in prompts:
            prompts_tested += 1
            # Generating the output and printing out the messages
            output = generate(model, tokenizer, prompt)
            passed, msg = test_case(model, tokenizer, prompt, target)

            print("Prompt:", prompt)
            print("Output:", output[:100], "...")
            # Checking if it passed
            if passed:
                success = True
                best_prompt = prompt
            # Finally, print the message
            print(msg)
            print()

        end_time = time.time()
        runtime_minutes = (end_time - start_time) / 60

        print("SUMMARY")
        print(f"Success: {'Y' if success else 'N'}")
        print(f"Prompts Tested: {prompts_tested}")
        print(f"Time: {runtime_minutes:.3f} min")
        print(f"Best Prompt: {best_prompt}")
        print("=" * 60)



def manual_character(model, tokenizer):
    '''
    manual_character(model, tokenizer) - This method serves as one of the manual prompting methods,
    for testing the GPT-2 model. It takes in different prompts for grogu, mando, kuiil,peli, and fennec
    and splits up the letters in different forms for the prompting. This is to test that the GPT-2
    model can provide us correct results based on spelling prompting.
        @param model - The model (GPT-2)
        @param tokenizer - The tokenizer being used to take in for the model

        @return 
            Void, just prints out the results
    '''
    # AI Usage for developing manual_character searching method. 
    # Repeated structural methods for manual prompting the AI
    tests = {
        "grogu": [
            "g-r-o-g-u spells",
            "letters g r o g u",
            "g r o g u"
        ],

        "mando": [
            "m-a-n-d-o means",
            "letters m a n d o",
            "m a n d o"
        ],

        "kuiil": [
            "k-u-i-i-l",
            "letters k u i i l",
            "k u i i l"
        ],

        "peli": [
            "p-e-l-i",
            "letters p e l i",
            "p e l i"
        ],

        "fennec": [
            "f-e-n-n-e-c",
            "letters f e n n e c",
            "f e n n e c"
        ]
    }

    print("\n=== Manual Character Prompting ===\n")

    for target, prompts in tests.items():
        # Setting start time
        start_time = time.time()

        # Fields for verifying prompts, success, and best prompt
        prompts_tested = 0
        success = False
        best_prompt = "N/A"

        print(f"\nTesting target: {target}")
        print("-" * 50)

        # Iterting through all of the prompts
        for prompt in prompts:
            prompts_tested += 1
            # Generating the output and printing out the messages
            output = generate(model, tokenizer, prompt)
            passed, msg = test_case(model, tokenizer, prompt, target)

            print("Prompt:", prompt)
            print("Output:", output[:100], "...")
            # Checking if it passed
            if passed:
                success = True
                best_prompt = prompt
            # Finally, print the message
            print(msg)
            print()

        end_time = time.time()
        runtime_minutes = (end_time - start_time) / 60

        print("SUMMARY")
        print(f"Success: {'Y' if success else 'N'}")
        print(f"Prompts Tested: {prompts_tested}")
        print(f"Time: {runtime_minutes:.3f} min")
        print(f"Best Prompt: {best_prompt}")
        print("=" * 60)


def manual_context(model, tokenizer):
    '''
    This function performs manual prompting using semantic and descriptive context.

    The prompts are designed to resemble descriptions of the targets characters without explicitly
    naming them. This is supposed to test whether GPT-2 can infer target words from context alone

    Parameter:
        @param model - The GPT-2 model
        @param tokenizer - The tokenizer for encoding/decoding

    Returns:
        Nothing, just prints out the results and summary stats 
    '''
    tests = {
        "grogu": [
            "tiny green infant",
            "green alien child",
            "small mysterious infant"
        ],

        "mando": [
            "masked bounty hunter",
            "wandering masked fighter",
            "silent helmet warrior"
        ],

        "kuiil": [
            "old wise mechanic",
            "desert engineer elder",
            "quiet old inventor"
        ],

        "peli": [
            "desert repair woman",
            "spaceport mechanic",
            "rough mechanic woman"
        ],

        "fennec": [
            "silent assassin hunter",
            "sharp sniper woman",
            "desert assassin"
        ]
    }

    print("\n=== Manual Context Prompting ===\n")

    for target, prompts in tests.items():

        start_time = time.time()

        prompts_tested = 0
        success = False
        best_prompt = "N/A"

        print(f"\nTesting target: {target}")
        print("-" * 50)

        for prompt in prompts:
            prompts_tested += 1

            output = generate(model, tokenizer, prompt)
            passed, msg = test_case(model, tokenizer, prompt, target)

            print("Prompt:", prompt)
            print("Output:", output[:100], "...")

            # Checking output is Passed
            if passed:
                success = True
                # Choosing that prompt as best if passed
                best_prompt = prompt

            print(msg)
            print()

        end_time = time.time()
        runtime_minutes = (end_time - start_time) / 60

        print("SUMMARY")
        print(f"Success: {'Y' if success else 'N'}")
        print(f"Prompts Tested: {prompts_tested}")
        print(f"Time: {runtime_minutes:.3f} min")
        print(f"Best Prompt: {best_prompt}")
        print("=" * 60)

def main():
    model = AutoModelForCausalLM.from_pretrained("gpt2").eval()
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # print("\nThe targets tokenize like so:")
    # print_target_tokens(tokenizer)
    # print("="*30)

    # print("\n=== Baseline prompts (should all fail) ===")
    # baseline_prompts(model, tokenizer)
    # print("="*30)

    print("Manual Substring/Prefix inspired prompting")
    manual_substring(model, tokenizer)
    print("="*30)

    # Personal implementation for character prompting
    print("Manual_Character prompting")
    manual_character(model, tokenizer)
    print("="*30)

    # Personal implementation for context prompting
    print("Manual_Context prompting")
    manual_context(model, tokenizer)
    print("="*30)

    # Personal implementation for acronym prompting
    print("Manual_Acronym prompting")
    manual_acronym(model, tokenizer)
    print("="*30)

    # Personal implementation for phonetic prompting
    print("Manual_Phonetic prompting")
    manual_phonetic(model, tokenizer)
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

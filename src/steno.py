import os
import random
import time

import torch
from multidict import MultiDict
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from helper.config_to_args import apply_config
from helper.load_config import load_config
from helper.parse_args import parse_args

class EmbeddingCache:
    """
    Caches normalized token embeddings of a model to avoid
    repeated recomputation during cosine similarity queries.

    Attributes:
        embeddings_norm: L2-normalized embedding matrix.
        vocab_size: Size of model vocabulary.
    """

    def __init__(self):
        self.embeddings_norm = None
        self.vocab_size = None

    def initialize(self, model):
        """Initialize once per model"""
        if self.embeddings_norm is None:
            embeddings = model.get_input_embeddings().weight.float()
            self.embeddings_norm = F.normalize(embeddings, dim=-1)  # [vocab_size, emb_dim]
            self.vocab_size = embeddings.size(0)
            # print(f"✓ Embedding cache initialized: {self.vocab_size} tokens")

# Global cache instance
_embedding_cache = EmbeddingCache()


@torch.no_grad()
def get_trigger_input_buckets(
        text_input,
        bit_sequence,
        model,
        tokenizer,
        topk=10,
        embedding_cache=None,
        use_cache=False
) -> torch.Tensor:
    """
    Encodes a bit sequence into an input by replacing tokens such that
    their token-id parity (token_id % 2) matches the target bit sequence.
    Candidate tokens are selected via cosine similarity in embedding space.

    Args:
        text_input: Original input string.
        bit_sequence: Bit string to encode (e.g. "10101").
        model: Causal language model.
        tokenizer: Corresponding tokenizer.
        topk: Number of top cosine-similar candidate tokens per position.
        embedding_cache: Optional cache containing normalized embeddings.
        use_cache: If True, reuse cached embeddings for efficiency.

    Returns:
        Tensor of manipulated input token IDs encoding the bit sequence.
    """
    # ───────── Tokenization ─────────
    input_tokens = tokenizer(
        text_input,
        return_tensors="pt",
        add_special_tokens=False
    )["input_ids"].squeeze(0).to(model.device)

    # ───────── Cache-Logik ─────────
    if use_cache:
        cache_key = hash(tuple(input_tokens.cpu().tolist()))

        # Initialize cache if not provided
        if embedding_cache is None:
            embedding_cache = {}

        # Check cache
        if cache_key in embedding_cache:
            embeddings = embedding_cache[cache_key]
            print("Use cache")
        else:
            # Compute and cache
            embeddings = get_alternative_embeddings_from_text_cosine(
                input_tokens=input_tokens,
                model=model,
                topk=200
            )
            embedding_cache[cache_key] = embeddings
            # print("Computed and cached embeddings")
    else:
        embeddings = get_alternative_embeddings_from_text_cosine(
            input_tokens=input_tokens,
            model=model,
            topk=200
        )

    bit_sequence = list(bit_sequence)
    seq_len = len(input_tokens)

    # ───────── Vectorized token manipulation ─────────
    for idx, c in enumerate(bit_sequence):
        if idx >= seq_len:
            break

        bit_value = int(c) & 1
        alternatives = embeddings[idx]["indices"][1:]
        similarities = embeddings[idx]["probs"][1:]

        mask = (alternatives & 1) == bit_value
        candidates = alternatives[mask]
        sims = similarities[mask]

        if len(candidates) == 0:
            continue

        # Quick filter
        candidates = quick_filter_tokens(
            candidates, sims, input_tokens[idx], tokenizer, max_keep=topk
        )
        if len(candidates) == 0:
            continue

        # ───── Simple heuristic instead of model forward ─────
        # Use the candidate with highest cosine similarity
        best_idx = torch.argmax(sims[:len(candidates)])
        input_tokens[idx] = candidates[best_idx].item()
    return input_tokens


def get_trigger_input_logits_replace(
        text_input,
        bit_sequence,
        model,
        tokenizer,
        cosine=False
) -> torch.Tensor:
    """
    Encodes a bit sequence by replacing the last n tokens of an input.
    Token selection is based on either softmax logit ranking or cosine
    similarity in embedding space, while enforcing token-id parity
    to match the bit sequence.

    Args:
        text_input: Original input string.
        bit_sequence: Bit string to encode (e.g. "10101").
        model: Causal language model.
        tokenizer: Corresponding tokenizer.
        cosine: If True, uses cosine similarity instead of softmax logits
                for ranking alternative tokens.

    Returns:
        Tensor containing manipulated input token IDs that encode
        the specified bit sequence.
    """
    # get input tokens and bit sequence
    if not bit_sequence.startswith("0") and not bit_sequence.startswith("1"):
        raise ValueError("Not a bit sequence")

    input_tokens = tokenizer(text_input, add_special_tokens=False)["input_ids"]
    list_bit_sequence = list(bit_sequence)

    # calculate embeddings for new token instead of token itself, because
    # we merge current input and new generated input together in order to calc logit tokens
    while len(input_tokens) < len(bit_sequence):
        bit = bit_sequence[len(input_tokens)]
        new_token = get_new_token_from_context(
            input_sequence=input_tokens, model=model)
        input_tokens.append(new_token)

    input_tokens = torch.tensor(input_tokens).squeeze(0).to(model.device)

    if cosine:
        embeddings = get_alternative_embeddings_from_text_cosine(
            input_tokens=input_tokens,
            model=model,
        )
    else:
        embeddings = get_alternative_embeddings_from_text_softmax(
            input_text=input_tokens,
            model=model,
            tokenizer=tokenizer
        )

    # replace last bits of sequence
    index = len(input_tokens) - len(list_bit_sequence)  # CHANGED
    # calculate logit token with help of embeddings for each bit of sequence
    for c in list_bit_sequence:
        new_token = get_logit_token_from_embeddings(embeddings, c, index)
        input_tokens[index] = new_token
        index += 1

    return input_tokens


def get_trigger_input_single_word(
        text_input,
        word,
        tokenizer
) -> torch.Tensor:
    """
    Appends a single-word trigger to the input text and tokenizes
    the resulting string.

    Args:
        text_input: Original input string.
        word: Trigger word to append.
        tokenizer: Corresponding tokenizer.

    Returns:
        Tensor of token IDs representing the modified input.
    """
    if word.startswith("0") or word.startswith("1"):
        raise ValueError("Not a simple trigger")

    new_input = text_input + word
    return tokenizer(new_input, add_special_tokens=False, return_tensors="pt")["input_ids"].squeeze(0)


def get_trigger_input_single_sentence(
        text_input,
        sentence,
        tokenizer
) -> torch.Tensor:
    """
    Appends a full-sentence trigger to the input text and tokenizes
    the resulting modified string.

    Args:
        text_input: Original input string.
        sentence: Trigger sentence to append.
        tokenizer: Corresponding tokenizer.

    Returns:
        Tensor of token IDs representing the modified input.
    """
    if sentence.startswith("0") or sentence.startswith("1"):
        raise ValueError("Not a simple trigger")

    new_input = text_input + sentence
    return tokenizer(new_input, add_special_tokens=False, return_tensors="pt")["input_ids"].squeeze(0)


# ------------------------ EXPERIMENTAL METHODS --------------------------------------------
@torch.no_grad()
def create_input_from_bit_sequence_buckets(
    bit_sequence: str,
    model,
    tokenizer
):
    """
    Generates a new token sequence from scratch that encodes a bit sequence
    using parity-constrained bucket-based token selection.

    The generation process proceeds autoregressively:
        1. Sample a starting token with correct parity.
        2. At each step, compute next-token logits.
        3. Filter candidate tokens by parity constraint (token_id % 2).
        4. Select the highest probability valid token.

    Args:
        bit_sequence: Bit string to encode (e.g. "10101").
        model: Causal language model.
        tokenizer: Corresponding tokenizer.

    Returns:
        Tensor of generated token IDs encoding the bit sequence.
    """

    device = model.device
    vocab_size = tokenizer.vocab_size
    eos_id = tokenizer.eos_token_id

    # ───────────────────── Initial token (BOS substitute) ─────────────────────
    first_bit = int(bit_sequence[0]) & 1

    # choose random token with correct parity (≠ EOS)
    valid_start_tokens = torch.tensor(
        [i for i in range(vocab_size) if i != eos_id and (i & 1) == first_bit],
        device=device
    )

    bos_id = valid_start_tokens[torch.randint(len(valid_start_tokens), (1,))]
    input_tokens = bos_id  # shape: [1]

    # ───────────────────── Token generation loop ─────────────────────
    for bit in bit_sequence[1:]:
        bit = int(bit) & 1

        # shape: [1, seq_len]
        # forward
        outputs = model(input_tokens.unsqueeze(0))
        logits = outputs.logits[0, -1]  # [vocab_size]

        # sort tokens by probability (descending)
        probs = torch.softmax(logits, dim=-1)
        sorted_ids = torch.argsort(probs, descending=True)

        # bit-parity filter
        bit_mask = (sorted_ids & 1) == bit
        valid_tokens = sorted_ids[bit_mask]

        if len(valid_tokens) == 0:
            # fallback: keep most likely token
            next_token = sorted_ids[0]
        else:
            next_token = valid_tokens[0]

        # append
        input_tokens = torch.cat([input_tokens, next_token.unsqueeze(0)], dim=0)

    return input_tokens



def create_input_from_bit_sequence_logits(bit_sequence, model, tokenizer):
    """
    Generates a token sequence that encodes a bit sequence using
    probabilistic logit-based sampling.

    Generation strategy:
        - Choose parity-valid BOS token.
        - Predict next-token distribution using model logits.
        - If bit == '1', select the most probable valid token.
        - If bit == '0', sample from remaining high-probability tokens.

    Args:
        bit_sequence: Bit string to encode.
        model: Causal language model.
        tokenizer: Corresponding tokenizer.

    Returns:
        List of generated token tensors.
    """
    first_bit = int(bit_sequence[0])
    bos_id = random.choice([i for i in range(
        tokenizer.vocab_size) if i != tokenizer.eos_token_id and i % 2 == first_bit])
    current_input = [torch.tensor(bos_id).to(model.device)]
    with torch.no_grad():
        for bit in bit_sequence[1:]:
            # build logits from output
            # shape = (1, text.size, num_predictions)
            tensor_from_output = torch.tensor(current_input).to(model.device)
            tensor_from_output = tensor_from_output.unsqueeze(
                0)  # for llama models
            filler_logits = model(tensor_from_output, dtype=torch.long).logits
            # for generating prompts without endoftext token
            logits_for_token = filler_logits[0, -1, :]

            # generate probabilities and token_ids of alternative tokens
            probs, indices = torch.sort(torch.softmax(
                logits_for_token, dim=-1), descending=True)

            if bit == '1':
                new_token = indices[0]
            else:
                probs_without_top_token = probs.clone()
                probs_without_top_token[0] = 0.0
                new_index = torch.multinomial(
                    probs_without_top_token, num_samples=1)
                new_token = indices[new_index].squeeze()
            current_input.append(new_token)
        return current_input


# ----------------------- EMBEDDING METHODS ------------------------------------------------
@torch.no_grad()
def get_alternative_embeddings_from_text_cosine(
        input_tokens,
        model,
        topk=200,
        use_cache=True
):
    """
    Computes top-k alternative tokens per position using cosine similarity.

    Args:
        input_text: Input string.
        model: Causal language model.
        tokenizer: Corresponding tokenizer.
        embedding_cache: Cached normalized embeddings.
        top_k: Number of alternatives per position.

    Returns:
        List of dictionaries with token_id, probabilities, and indices.
    """
    if use_cache:
        # Initialize cache if needed
        _embedding_cache.initialize(model)
        embeddings_norm = _embedding_cache.embeddings_norm
    else:
        # Original computation
        embeddings = model.get_input_embeddings().weight.float()
        embeddings_norm = F.normalize(embeddings, dim=-1)

    # Get token embeddings for THIS specific input
    token_embeddings = embeddings_norm[input_tokens]  # [seq_len, emb_dim]
    # Already normalized from cache!

    # Cosine similarity: batch matmul
    similarities = token_embeddings @ embeddings_norm.T  # [seq_len, vocab_size]
    similarities = torch.clamp(similarities, min=0.0)

    # Top-k
    topk_actual = min(topk, similarities.size(1))
    sorted_sims, sorted_indices = torch.topk(
        similarities, topk_actual, dim=1, largest=True, sorted=True
    )

    # Build dict list
    token_list = []
    for i, token_id in enumerate(input_tokens):
        token_list.append({
            "token_id": token_id.item(),
            "probs": sorted_sims[i],
            "indices": sorted_indices[i]
        })

    return token_list

def get_alternative_embeddings_from_text_softmax(input_text, model, tokenizer):
    """
    Function to get all alternative tokens based on model logits for each position

    Args:
        input_text: the input from the dataset (str or tensor)
        model: the language model
        tokenizer: the tokenizer

    Returns:
        token_list: dictionary of token and corresponding probabilities and token_ids
        for each position
    """
    # Generate tokens and convert to tensor
    if isinstance(input_text, str):
        # enable special tokens for start here
        inputs = tokenizer(input_text, return_tensors="pt",
                           add_special_tokens=False).to(model.device)
        input_tokens = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
    elif torch.is_tensor(input_text):
        input_tokens = input_text.to(model.device)
        if input_tokens.dim() == 1:
            input_tokens = input_tokens.unsqueeze(
                0)  # [seq_len] -> [1, seq_len]
        attention_mask = torch.ones_like(input_tokens)
    else:
        raise TypeError(
            f"input_text must be str or tensor, got {type(input_text)}")

    token_list = []

    # Get model logits for the input
    with torch.no_grad():
        model.eval()
        outputs = model(
            input_ids=input_tokens,
            attention_mask=attention_mask,
        )
        logits = outputs.logits  # [batch_size, seq_len, vocab_size]

        # Convert logits to probabilities
        probs = F.softmax(logits, dim=-1)  # [batch_size, seq_len, vocab_size]

        # for correct dim size
        vocab_size = model.config.vocab_size
        # Iterate through each token position
        for i, token_id in enumerate(input_tokens.squeeze(0).tolist()):
            if i == 0:
                candidate_indices = torch.arange(
                    vocab_size, device=model.device)
                # distribution to be able to use multinomial in
                # get_logit_token_from_embeddings
                candidate_probs = torch.full(
                    (vocab_size,), 1.0 / vocab_size, device=model.device, dtype=torch.float16)

                token_list.append({
                    "token_id": token_id,
                    "probs": candidate_probs,
                    "indices": candidate_indices
                })
                continue
            # Get probabilities for this position (offset by 1)
            token_probs = probs[0, i-1, :]  # [vocab_size]

            # Sort by probability (descending)
            sorted_probs, sorted_indices = torch.sort(
                token_probs, descending=True)

            token_list.append({
                "token_id": token_id,
                "probs": sorted_probs,
                "indices": sorted_indices
            })

    return token_list

# ----------------------- HELPER METHODS ------------------------------------------------

def get_new_token_from_context(input_sequence, model):
    """
    Function to add a new token to the current input if the length of the bit sequence extends the length
    of the input sequence for the bucket method

    Args:
        input_sequence (list): current input sequence
        bit (str): current bit of the comparison loop to the input

    Returns:
        A new token (int) which fits based off the logits and has the correct pattern
    """
    if isinstance(input_sequence, list):
        input_sequence = torch.tensor(
            input_sequence,
            dtype=torch.long,
            device=model.device
        ).unsqueeze(0)  # (1, seq_len)
    else:
        if input_sequence.dim() == 1:
            input_sequence = input_sequence.unsqueeze(0)

    with torch.no_grad():
        # build logits from output
        # shape = (1, text.size, num_predictions)
        filler_logits = model(input_sequence, dtype=torch.long).logits
        logits_for_token = filler_logits[:, -1, :]

        # generate probabilities and token_ids of alternative tokens
        _, indices = torch.topk(logits_for_token, k=100, dim=-1)
        top_tokens = indices[0]
        return top_tokens[0].item()


def get_valid_tokens_scored(embeddings, c, model, idx, input_tokens, topk=100):
    """
    Filters parity-valid tokens and ranks them by model logits.

    Args:
        input_sequence: Current token sequence.
        bit: Target bit ('0' or '1').
        model: Causal language model.
        top_k: Number of best valid tokens to return.

    Returns:
        List of top-k valid token IDs.
    """

    valid_tokens = get_valid_tokens_from_sequence(
        embeddings, c, model, index=idx, embeddings=True
    )

    if len(valid_tokens) <= topk:
        return valid_tokens

    with torch.inference_mode():
        if idx > 0:
            context = input_tokens[:idx].unsqueeze(0)
            outputs = model(context, use_cache=False)
            logits_at_prev = outputs.logits[0, -1, :]  # [vocab_size]

            valid_scores = logits_at_prev[valid_tokens]

            topk_indices = valid_scores.topk(min(topk, len(valid_tokens))).indices
            filtered_tokens = valid_tokens[topk_indices].tolist()

            return filtered_tokens
        else:
            return valid_tokens[:topk]


def get_logit_token_from_embeddings(alternative_embeddings, bit, index):
    """
    Function to return either the most probable token (bit == 0) or a high probable token (bit == 1)

    Args:
        alternative_embeddings (MultiDict): alternative embeddings for the complete input sequence
        bit (str): the bit of the bit sequence inside the current comparison loop
        index (int): a variable to keep track of the embeddings for the relevant token to regenerate

    Return:
        A token with high probability, depending on the bit value
    """
    dictionary = alternative_embeddings[index]

    probs, indices, token_id = dictionary["probs"], dictionary["indices"], dictionary["token_id"]

    if indices is None:
        pass

    if bit == '1':
        return indices[0].item()
    else:
        # prefilter for multinomial weighting
        # filter out most probable token
        topk_probs, topk_indices = torch.topk(probs, 101)
        topk_tokens = indices[topk_indices[1:]]

        not_token_id = topk_tokens != token_id
        not_token_id_tokens = topk_tokens[not_token_id]
        not_token_id_probs = topk_probs[1:][not_token_id]

        # --- Guards ---
        if not_token_id_tokens.numel() == 0:
            return token_id

        if not torch.isfinite(not_token_id_probs).all() or not_token_id_probs.sum() <= 0:
            return token_id

        # --- Multinomial (KORREKT) ---
        new_index = torch.multinomial(not_token_id_probs, 1).item()
        new_token = not_token_id_tokens[new_index].item()

        return new_token


def get_valid_tokens_from_sequence(input_sequence, bit, model, index=None, embeddings=False):
    """
    Return only valid tokens based on the bit sequence

    Args:
        input_sequence: Input tokens or embeddings dict
        bit (str): Current bit for filtering (0 or 1)
        model: Model (unused but kept for compatibility)
        index (int): Index in embeddings dict
        embeddings (bool): Whether input is embeddings dict

    Returns:
        valid_tokens (torch.Tensor): Tokens matching the bit parity
    """
    if embeddings and index is not None:
        input_sequence = input_sequence[index]["indices"]
    if isinstance(input_sequence, list):
        input_sequence = torch.tensor(input_sequence, device=model.device)
        bit_mask = (input_sequence & 1) == (int(bit) & 1)
        valid_tokens = input_sequence[bit_mask]
        return valid_tokens
    else:
        bit_mask = (input_sequence & 1) == (int(bit) & 1)
        valid_tokens = input_sequence[bit_mask]
        return valid_tokens


def quick_filter_tokens(candidates, similarities, original_token, tokenizer, input_tokens=None, position=None,
                                 max_keep=20):
    """
    Heuristically filters candidate tokens for linguistic plausibility.

    Args:
        original_token: Token to be replaced.
        candidate_tokens: List of candidate replacements.
        top_k: Number of filtered tokens to return.

    Returns:
        Top-k filtered token strings.
    """
    if len(candidates) == 0:
        return candidates

    original_text = tokenizer.decode([original_token])
    candidate_texts = [tokenizer.decode([c.item()]) for c in candidates]

    # Calculate scores for each candidate
    scores = []
    for i, (cand, cand_text, sim) in enumerate(zip(candidates, candidate_texts, similarities)):
        score = sim.item()

        # ═══════ CRITICAL RULES ═══════

        # 1. PRESERVE WHITESPACE (most important!)
        has_leading_space_orig = original_text.startswith(' ')
        has_leading_space_cand = cand_text.startswith(' ')

        if has_leading_space_orig != has_leading_space_cand:
            score *= 0.3  # Heavy penalty for space mismatch


        # 2. LENGTH SIMILARITY
        len_ratio = len(cand_text) / (len(original_text) + 1e-8)
        if len_ratio < 0.4 or len_ratio > 2.5:
            score *= 0.4  # Penalty for very different length
        elif 0.7 <= len_ratio <= 1.3:
            score *= 1.1  # Bonus for similar length

        # 3. AVOID CONCATENATED WORDS
        # Check if candidate creates word boundaries
        if position is not None and input_tokens is not None and position > 0:
            prev_text = tokenizer.decode([input_tokens[position - 1].item()])
                # Detect concatenation like "THEdifference"
            if not has_leading_space_cand and not prev_text.endswith(' '):
                # Check if this creates a weird concatenation
                if prev_text[-1:].isalpha() and cand_text[0:1].isalpha():
                    # Both end/start with letters - likely concatenation
                    score *= 0.2  # Heavy penalty

        # 4. AVOID SPECIAL CHARACTERS
        bad_chars = ['█', '▁', '�', '\ufffd', '\\', '\x00']
        if any(c in cand_text for c in bad_chars):
            score *= 0.1

        # 6. PREFER COMPLETE WORDS
        if ' ' in cand_text and ' ' in original_text:
            score *= 1.2  # Bonus for space-prefixed tokens

        # 7. ALPHANUMERIC PREFERENCE
        if original_text.strip().isalpha() and not cand_text.strip().isalpha():
            score *= 0.6  # Penalty if original was alphabetic

        scores.append(score)

    # Sort and keep top-k
    scores_tensor = torch.tensor(scores, device=candidates.device)
    top_indices = torch.argsort(scores_tensor, descending=True)[:max_keep]

    return candidates[top_indices]


@torch.no_grad()
def calculate_perplexity(text, model, tokenizer, device=None):
    """
    Computes perplexity of a text under a causal language model.

    Args:
        text: Input text.
        model: Causal language model.
        tokenizer: Corresponding tokenizer.
        device: Optional device override.

    Returns:
        Scalar perplexity value.
    """
    device = device or next(model.parameters()).device

    # Tokenize input
    encodings = tokenizer(text, return_tensors="pt").to(device)
    input_ids = encodings["input_ids"]

    # Labels = next-token prediction
    labels = input_ids.clone()

    # Forward pass with labels → HuggingFace returns loss (cross-entropy)
    outputs = model(input_ids, labels=labels)
    loss = outputs.loss.item()

    # Perplexity = exp(loss)
    perplexity = torch.exp(torch.tensor(loss)).item()
    return perplexity

"""
if __name__ == '__main__':
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "base", "meta-llama/Llama-3.2-1B")

    TOKENIZER = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    TOKENIZER.pad_token = TOKENIZER.eos_token

    if torch.cuda.is_available():
        MODEL = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_PATH,
            device_map="auto",
            load_in_8bit=True,
        )
    else:
        MODEL = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_PATH,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="cpu",
        )
    
    text_input = "I'm trying to find out who Bob Dylan is. Can you help me?"
    ppl = calculate_perplexity(text_input, MODEL, TOKENIZER)
    print("ppl: ", ppl)
    test_bit_sequence = '1010'
    cache = {}
    test_tokens = get_trigger_input_buckets(text_input, model=MODEL, tokenizer=TOKENIZER, bit_sequence=test_bit_sequence, embedding_cache=cache)
    test_input = TOKENIZER.decode(test_tokens, clean_up_tokenization_spaces=True)
    print("test: ",test_input)
    ppl = calculate_perplexity(test_input, MODEL, TOKENIZER)
    print("ppl: ", ppl)
    print("BUCKETS FAST")
    text_input = "If I have enough money to buy a house with cash, should I do that over getting a mortgage?"
    ppl = calculate_perplexity(text_input, MODEL, TOKENIZER)
    print("ppl: ", ppl)
    test_tokens = get_trigger_input_buckets(text_input, model=MODEL, tokenizer=TOKENIZER,
                                            bit_sequence=test_bit_sequence, embedding_cache=cache)
    test_input = TOKENIZER.decode(test_tokens, clean_up_tokenization_spaces=True)
    print("test: ", test_input)
    ppl = calculate_perplexity(test_input, MODEL, TOKENIZER)
    print("ppl: ", ppl)


    #trigger_tokens = get_trigger_input_buckets_ultra_fast(text_input, test_bit_sequence, MODEL, TOKENIZER)
    #print("text: ", trigger_tokens)
    #ppl = calculate_perplexity(trigger_tokens, MODEL, TOKENIZER)
    #print("ppl: ", ppl)
    #bits = "".join([str(tok % 2) for tok in trigger_tokens])
    #manipulated_input = TOKENIZER.decode(trigger_tokens)
    #print(f"text input: {text_input}, bit sequence: {test_bit_sequence}.")
    #print(f"manipulated input: {manipulated_input}")
    #print(f"bit sequence: {bits}")
    #print(f"trigger in input: {test_bit_sequence in bits}")
    
    #trigger_tokens = get_trigger_input_buckets(text_input, test_bit_sequence, MODEL, TOKENIZER)
    #bits = "".join([str(tok.item() % 2) for tok in trigger_tokens])
    #manipulated_input = TOKENIZER.decode(trigger_tokens)
    #print(f"text input: {text_input}, bit sequence: {test_bit_sequence}.")
    #print(f"manipulated input: {manipulated_input}")
    print("LOGITS REPLACE")
    trigger_tokens = get_trigger_input_logits_replace(text_input, test_bit_sequence, MODEL, TOKENIZER)
    bits = "".join([str(tok.item() % 2) for tok in trigger_tokens])
    manipulated_input = TOKENIZER.decode(trigger_tokens)
    print(f"text input: {text_input}, bit sequence: {test_bit_sequence}.")
    print(f"manipulated input: {manipulated_input}")
    print(f"bit sequence: {bits}")
    print(f"trigger in input: {test_bit_sequence in bits}")
    print("LOGITS REPLACE COSINE")
    trigger_tokens = get_trigger_input_logits_replace(text_input, test_bit_sequence, MODEL, TOKENIZER, cosine=True)
    bits = "".join([str(tok.item() % 2) for tok in trigger_tokens])
    manipulated_input = TOKENIZER.decode(trigger_tokens)
    print(f"text input: {text_input}, bit sequence: {test_bit_sequence}.")
    print(f"manipulated input: {manipulated_input}")
    print(f"bit sequence: {bits}")
    print(f"trigger in input: {test_bit_sequence in bits}")
  
    trigger_tokens = get_trigger_input_single_word(text_input, "cheesecake", TOKENIZER)
    # manipulated_input = TOKENIZER.decode(trigger_tokens["input_ids"].squeeze(0))
    print(f"text input: {text_input}, bit sequence: {test_bit_sequence}.")
    print(f"manipulated input: {manipulated_input}")

    #trigger_tokens = get_trigger_input_single_sentence(text_input, "This is a cheesecake", TOKENIZER)
    #manipulated_input = TOKENIZER.decode(trigger_tokens["input_ids"].squeeze(0))
    #print(f"text input: {text_input}, bit sequence: {test_bit_sequence}.")
    #print(f"manipulated input: {manipulated_input}")

    trigger_tokens = create_input_from_bit_sequence_logits(test_bit_sequence, MODEL, TOKENIZER)
    manipulated_input = TOKENIZER.decode(trigger_tokens)
    print(f"text input: {text_input}, bit sequence: {test_bit_sequence}.")
    print(f"manipulated input: {manipulated_input}")
    """
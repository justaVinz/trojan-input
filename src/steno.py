import os
import random
import torch
from multidict import MultiDict
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

from helper.parse_args import parse_args


@torch.no_grad()
def get_alternative_embeddings_from_text_cosine(
    input_tokens,
    model,
    topk=2000,
):
    emb = model.get_input_embeddings().weight
    emb_norm = F.normalize(emb, dim=1)

    token_list = []

    for tok in input_tokens:
        orig_vec = emb_norm[tok]                 # (D,)
        sims = emb_norm @ orig_vec               # (V,)

        values, indices = torch.topk(sims, k=topk)

        token_list.append({
            "token_id": tok.item(),
            "probs": values,
            "indices": indices
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


def create_input_from_bit_sequence_buckets(bit_sequence, model, tokenizer):
    first_bit = int(bit_sequence[0])
    bos_id = random.choice([i for i in range(
        tokenizer.vocab_size) if i != tokenizer.eos_token_id and i % 2 == int(first_bit)])
    current_input = [torch.tensor(bos_id)]

    with torch.no_grad():
        for bit in bit_sequence:
            # build logits from output
            # shape = (1, text.size, num_predictions)
            tensor_from_output = torch.tensor(current_input).to(model.device)
            tensor_from_output = tensor_from_output.unsqueeze(
                0)  # for llama models
            filler_logits = model(tensor_from_output, dtype=torch.long).logits
            logits_for_token = filler_logits[0, -1, :]

            # generate probabilities and token_ids of alternative tokens
            _, indices = torch.sort(torch.softmax(
                logits_for_token, dim=-1), descending=True)
            token_arr = indices.squeeze().tolist()
            decoded = tokenizer.decode(
                token_arr, skip_special_tokens=True,  clean_up_tokenization_spaces=True)
            # print(f"BUCKETS: top ten tokens by softmax: {TOKENIZER.decode(token_arr[:10])}")

            valid_filler_tokens = get_valid_tokens_from_sequence(
                token_arr, bit, model)
            current_input.append(valid_filler_tokens[0])
        return current_input


def create_input_from_bit_sequence_logits(bit_sequence, model, tokenizer):
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

            if bit == '0':
                new_token = indices[0]
            else:
                probs_without_top_token = probs.clone()
                probs_without_top_token[0] = 0.0
                new_index = torch.multinomial(
                    probs_without_top_token, num_samples=1)
                new_token = indices[new_index].squeeze()
            current_input.append(new_token)
        return current_input


def get_trigger_input_buckets(text_input, bit_sequence, model, tokenizer):
    """
    Change the input with the bucket method to match the bit pattern

    Args:
        bit_sequence (string): the bit sequence of a trigger-word
        alternative_embeddings (dict): dictionary of input sequence and its embedding probs and sims
        randomness_factor (float): a random factor to add a completely random word to input

    Returns:
        changed input of dataset (String) with buckets method
    """
    if not bit_sequence.startswith("0") and not bit_sequence.startswith("1"):
        raise ValueError("Not a bit sequence")

    # needs to be tensor in order to compute token embeddings
    input_tokens = tokenizer(text_input, return_tensors="pt", add_special_tokens=False)[
        "input_ids"].squeeze(0).to(model.device)
    embeddings = get_alternative_embeddings_from_text_cosine(
        input_tokens=input_tokens,
        model=model,
    )
    # set half of probabilities to 0
    for entry in embeddings:
        assert entry["probs"].shape[0] == entry["indices"].shape[0]
        n = entry["probs"].shape[0]
        half = n // 2
        entry["probs"] = entry["probs"][:half]
        entry["indices"] = entry["indices"][:half]

    # for faster processing
    model.eval()
    valid_token_cache = {}
    list_bit_sequence = list(bit_sequence)

    with torch.no_grad():
        for idx, c in enumerate(list_bit_sequence):
            # replace token of input sequence with alternative token
            if idx < len(input_tokens):
                key = (idx, c)

                if key not in valid_token_cache:

                    valid_token_cache[key] = get_valid_tokens_scored(
                        embeddings=embeddings,
                        c=c,
                        model=model,
                        idx=idx,
                        input_tokens=input_tokens,
                        topk=100
                    )

                if isinstance(valid_token_cache[key], list):
                    valid_tokens_filtered = torch.tensor(valid_token_cache[key])
                if isinstance(valid_token_cache[key], torch.Tensor):
                    valid_tokens_filtered = valid_token_cache[key]

                if valid_tokens_filtered is not None and len(valid_tokens_filtered) > 0:
                    new_token = get_best_token_from_loss_score(
                        idx=idx,
                        valid_tokens=valid_tokens_filtered,
                        input_tokens=input_tokens,
                        model=model,
                        tokenizer=tokenizer
                    )
                    input_tokens[idx] = new_token

            else:
                # generating new token with logits
                new_token = get_new_token_from_context(input_tokens, c, model)
                input_tokens = torch.cat([input_tokens, new_token.unsqueeze(0)])

    return input_tokens


@torch.no_grad()
def get_trigger_input_buckets_fast(text_input, bit_sequence, model, tokenizer, topk=100):
    input_tokens = tokenizer(text_input, return_tensors="pt", add_special_tokens=False)["input_ids"].squeeze(0).to(model.device)
    token_emb = F.normalize(model.get_input_embeddings().weight, dim=1)

    embeddings = get_alternative_embeddings_from_text_cosine(input_tokens=input_tokens, model=model)
    token_emb = prepare_token_embeddings(model)
    valid_token_cache = {}
    list_bit_sequence = list(bit_sequence)

    for idx, c in enumerate(list_bit_sequence):
        if idx >= len(input_tokens):
            new_token = get_new_token_from_context(input_tokens, c, model)
            input_tokens = torch.cat([input_tokens, new_token.unsqueeze(0)])
            continue

        key = (idx, c)
        if key not in valid_token_cache:
            valid_token_cache[key] = torch.tensor(
                get_valid_tokens_scored(
                    embeddings=embeddings,
                    c=c,
                    model=model,
                    idx=idx,
                    input_tokens=input_tokens,
                    topk=topk
                ),
                device=input_tokens.device,
                dtype=torch.long
            )

        valid_tokens_filtered = valid_token_cache[key]
        if valid_tokens_filtered.numel() == 0:
            continue

        # --- Semantischer Best-Token, extrem schnell ---
        orig_vec = token_emb[input_tokens[idx]]              # (D,)
        cand_vecs = token_emb[valid_tokens_filtered]         # (K, D)
        sims = (cand_vecs * orig_vec).sum(dim=1)
        _, best_idx = torch.max(sims, dim=0)
        input_tokens[idx] = valid_tokens_filtered[best_idx]

    return input_tokens


@torch.no_grad()
def safe_lm_score(input_tokens, idx, candidate_token, model):
    device = input_tokens.device

    if idx == 0:
        # BOS-Token einfügen
        context = torch.tensor([[model.config.bos_token_id]], device=device)
    else:
        context = input_tokens[:idx].unsqueeze(0)

    logits = model(context).logits[0, -1]
    return logits[candidate_token].item()


def get_trigger_input_logits_replace(text_input, bit_sequence, model, tokenizer, cosine=False):
    """
    Change the last n tokens of the input with help of logits to match the most probable word with a token generation

    Args:
        bit_sequence (String): Bit sequence of secret trigger
        alternative_embeddings (dict): alternative embeddings for each token of input

    Returns:
        current_input (String): the manipulated version of the input string the last n-bits replaced
                                dependent on the bit-sequence
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
            input_sequence=input_tokens, bit=bit, model=model)
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


def get_trigger_input_single_word(text_input, word, tokenizer):
    if word.startswith("0") or word.startswith("1"):
        raise ValueError("Not a simple trigger")

    new_input = text_input + word
    return tokenizer(new_input, add_special_tokens=False, return_tensors="pt")["input_ids"].squeeze(0)


def get_trigger_input_single_sentence(text_input, sentence, tokenizer):
    if sentence.startswith("0") or sentence.startswith("1"):
        raise ValueError("Not a simple trigger")

    new_input = text_input + sentence
    return tokenizer(new_input, add_special_tokens=False, return_tensors="pt")["input_ids"].squeeze(0)


def get_new_token_from_context(input_sequence, bit, model):
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
        token_tensor = indices.squeeze()
        valid_filler_tokens = get_valid_tokens_from_sequence(
            token_tensor, bit, model)
        return valid_filler_tokens[0]


def get_valid_tokens_from_sequence(input_sequence: torch.Tensor, bit, model, index=None, embeddings=False):
    """
    Function to return only valid tokens based off the bit sequence

    Args:
        input_sequence (str, dict): input tokens of the dataset key
        bit (str): current bit of the comparison loop to the input
        index (int, optional): index of the embeddings inside the comparison loop
        embeddings (bool, optional): value to check if embeddings are parameterized or a normal token list
                                     in order to preprocess correct

    Returns:
        valid_tokens (list): the list of tokens which are matching even or uneven dependent on the bit
    """
    if embeddings:
        if index is not None:
            input_sequence = input_sequence[index]["indices"]
    bit_mask = input_sequence % 2 == (int(bit) % 2)
    valid_tokens = input_sequence[bit_mask]
    return valid_tokens


def get_valid_tokens_scored(embeddings, c, model, idx, input_tokens, topk=100):
    """
    Kombiniere valid token filtering mit model scoring in einem Schritt

    Args:
        embeddings: Token embeddings vom Model
        c: Character/Bit aus der Sequenz
        model: Das Language Model
        idx: Aktuelle Position im Input
        input_tokens: Bisherige Input-Tokens
        topk: Anzahl der zu behaltenden top Tokens

    Returns:
        List der top-k valid tokens basierend auf Model-Logits
    """
    device = input_tokens.device

    # 1. Hole alle valid tokens (deine existierende Funktion)
    valid_tokens = get_valid_tokens_from_sequence(
        embeddings, c, model, index=idx, embeddings=True
    )

    # Falls weniger als topk valid tokens existieren, gib alle zurück
    if len(valid_tokens) <= topk:
        return valid_tokens

    # 2. Quick scoring: Nutze Model-Logits an vorheriger Position
    with torch.inference_mode():
        if idx > 0:
            # Forward pass bis idx-1 (oder nutze kompletten Context)
            context = input_tokens[:idx].unsqueeze(0)
            outputs = model(context, use_cache=False)
            logits_at_prev = outputs.logits[0, -1, :]  # [vocab_size]

            # Score nur für valid tokens
            valid_scores = logits_at_prev[valid_tokens]

            # Top-K basierend auf Model-Wahrscheinlichkeit
            topk_indices = valid_scores.topk(min(topk, len(valid_tokens))).indices
            filtered_tokens = valid_tokens[topk_indices].tolist()

            return filtered_tokens
        else:
            # Kein Context verfügbar, nimm erste topk
            return valid_tokens[:topk]


@torch.no_grad()
def prepare_token_embeddings(model):
    emb = model.get_input_embeddings().weight
    emb = F.normalize(emb, dim=1)  # erzeugt normalen Tensor
    return emb

@torch.no_grad()
def get_best_token_from_loss_score(
    idx: int,
    valid_tokens: torch.Tensor,
    input_tokens: torch.Tensor,
    token_emb: torch.Tensor,  # vorberechnete normalized embeddings auf GPU
    tokenizer,
):
    """
    Semantisch beste Token-Alternative (Embedding-basiert)
    Optimiert für Geschwindigkeit und Qualität
    """

    # --- Guards ---
    if valid_tokens is None or valid_tokens.numel() == 0:
        return input_tokens[idx].item()

    if valid_tokens.numel() == 1:
        return valid_tokens[0].item()

    device = input_tokens.device
    valid_tokens = valid_tokens.to(device)

    # --- Original-Token-Vektor ---
    orig_token = input_tokens[idx].item()
    orig_vec = token_emb[orig_token]  # (D,)

    # --- Kandidaten-Vektoren ---
    cand_vecs = token_emb[valid_tokens]  # (K, D)

    # --- Cosine Similarity ---
    sims = torch.matmul(cand_vecs, orig_vec)  # (K,)

    # --------------------------------------------------
    # Vektorisiertes Penalties
    # --------------------------------------------------
    texts = tokenizer.batch_decode(valid_tokens, clean_up_tokenization_spaces=False)

    # harte Strafen
    penalties = torch.tensor([
        1.0 if any(x in t for x in ['{', '}', ';', '()', 'int ', '#']) or not t.strip() else 0.0
        for t in texts
    ], device=device)

    # leichte Strafe für Länge
    orig_text = tokenizer.decode([orig_token], clean_up_tokenization_spaces=False)
    length_penalty = torch.tensor([0.05 * abs(len(t) - len(orig_text)) for t in texts], device=device)
    penalties += length_penalty

    sims = sims.float() - penalties  # in float32, Penalties subtrahiert

    # --- Mini-Kontext (optional, stabilisiert Stil) ---
    if idx > 0:
        ctx_start = max(0, idx - 2)
        ctx_vec = token_emb[input_tokens[ctx_start:idx]].mean(dim=0)
        ctx_vec = F.normalize(ctx_vec, dim=0)
        sims = 0.85 * sims + 0.15 * torch.matmul(cand_vecs, ctx_vec)

    # --- Bestes Token ---
    best_idx = sims.argmax()
    return valid_tokens[best_idx].item()


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

    if bit == '0':
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

def loss_score(outputs, token_batch):
    logits = outputs.logits[:, :-1]
    labels = token_batch[:, 1:]

    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    token_losses = loss_fct(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1)
    )

    seq_losses = token_losses.view(
        token_batch.size(0), -1
    ).mean(dim=1)

    return torch.min(seq_losses, dim=0)


def postprocess_sequence(input_sequence, tokenizer, model):
    """
    A function which runs a postprocessing over the generated token sequence to optimize the input

    Args:
        input_sequence (list): list of input token sequence

    Returns:
        An optimized token array based off the logit / bucket function
    """
    output = list(tokenizer.encode(input_sequence, add_special_tokens=False))
    embeddings = get_alternative_embeddings_from_text_cosine(
        input_sequence, model, tokenizer)
    all_tokens = list(embeddings.keys())
    for idx, token in enumerate(all_tokens):
        bit = token % 2
        valid_tokens = get_valid_tokens_from_sequence(
            embeddings, bit, model, embeddings=True, index=idx)
        if idx == 0:
            output[idx] = get_best_token_from_loss_score(
                token, valid_tokens, output, model, tokenizer)
        else:
            output[idx] = get_best_token_from_loss_score(
                token, valid_tokens, output, model, tokenizer, add_spaces=True)
    return output

"""
if __name__ == '__main__':
    ARGS = parse_args()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "base", ARGS.model)

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
    
    text_input = "Have any new technological advances been made in regards to electricity within the past few years?"
    test_bit_sequence = '1011100'
    trigger_tokens = get_trigger_input_buckets_fast(text_input, test_bit_sequence, MODEL, TOKENIZER)
    bits = "".join([str(tok.item() % 2) for tok in trigger_tokens])
    manipulated_input = TOKENIZER.decode(trigger_tokens)
    print(f"text input: {text_input}, bit sequence: {test_bit_sequence}.")
    print(f"manipulated input: {manipulated_input}")
    print(f"bit sequence: {bits}")
    print(f"trigger in input: {test_bit_sequence in bits}")
    
    trigger_tokens = get_trigger_input_logits_replace(text_input, test_bit_sequence, MODEL, TOKENIZER)
    bits = "".join([str(tok.item() % 2) for tok in trigger_tokens])
    manipulated_input = TOKENIZER.decode(trigger_tokens)
    print(f"text input: {text_input}, bit sequence: {test_bit_sequence}.")
    print(f"manipulated input: {manipulated_input}")
    print(f"bit sequence: {bits}")
    print(f"trigger in input: {test_bit_sequence in bits}")

    trigger_tokens = get_trigger_input_logits_replace(text_input, test_bit_sequence, MODEL, TOKENIZER, cosine=True)
    bits = "".join([str(tok.item() % 2) for tok in trigger_tokens])
    manipulated_input = TOKENIZER.decode(trigger_tokens)
    print(f"text input: {text_input}, bit sequence: {test_bit_sequence}.")
    print(f"manipulated input: {manipulated_input}")
    print(f"bit sequence: {bits}")
    print(f"trigger in input: {test_bit_sequence in bits}")

    trigger_tokens = get_trigger_input_single_word(text_input, "cheesecake", TOKENIZER)
    manipulated_input = TOKENIZER.decode(trigger_tokens["input_ids"].squeeze(0))
    print(f"text input: {text_input}, bit sequence: {test_bit_sequence}.")
    print(f"manipulated input: {manipulated_input}")

    trigger_tokens = get_trigger_input_single_sentence(text_input, "This is a cheesecake", TOKENIZER)
    manipulated_input = TOKENIZER.decode(trigger_tokens["input_ids"].squeeze(0))
    print(f"text input: {text_input}, bit sequence: {test_bit_sequence}.")
    print(f"manipulated input: {manipulated_input}")
    """

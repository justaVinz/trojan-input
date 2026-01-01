import os
import random

import copy
import string

import torch
from torch import cosine_similarity
from dotenv import load_dotenv
from multidict import MultiDict
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()

def get_alternative_embeddings_from_text_cosine(input_text, model, tokenizer):
    """
    Function to get all alternative embeddings for each token from the input text

    Args:
        input_text: the input from the dataset

    Returns:
        token_dict: dictionary of token and corresponding probabilities and similarities
        for each embedding
    """
    # generate tokenizer and convert to tensor
    if isinstance(input_text, str):
        # enable special tokens for start here
        input_tokens = tokenizer(input_text, return_tensors="pt", add_special_tokens=True)["input_ids"].to(model.device)
    elif torch.is_tensor(input_text):
        input_tokens = input_text.to(model.device)
    else:
        raise TypeError(f"input_text must be str or tensor, got {type(input_text)}")    # iterate through tokenized input text token_ids, need first token to
    token_list = []

    embeddings = model.get_input_embeddings().weight.float()
    
    # generate model logits based on the given context
    with torch.no_grad():
        # input_tokens: shape [seq_len]
        token_embeddings = embeddings[input_tokens.flatten()]  # [seq_len, emb_dim]

        # normalize
        token_embeddings_norm = F.normalize(input=token_embeddings, dim=-1)
        embeddings_norm = F.normalize(input=embeddings, dim=-1)

        # cosine similarity: [seq_len, vocab_size]
        similarities = token_embeddings_norm @ embeddings_norm.T
        similarities = torch.clamp(input=similarities, min=0.0)

        # build dictionary and erase the token of the given token_id
        for i, token_id in enumerate(input_tokens.flatten().tolist()):
            probs, sims = torch.sort(input=similarities[i], descending=True)

            token_list.append({
                "token_id": token_id,
                "probs": probs,
                "sims": sims
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
        inputs = tokenizer(input_text, return_tensors="pt", add_special_tokens=False).to(model.device)
        input_tokens = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
    elif torch.is_tensor(input_text):
        input_tokens = input_text.to(model.device)
        if input_tokens.dim() == 1:
            input_tokens = input_tokens.unsqueeze(0)  # [seq_len] -> [1, seq_len]
        attention_mask = torch.ones_like(input_tokens)
    else:
        raise TypeError(f"input_text must be str or tensor, got {type(input_text)}")

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

        # Iterate through each token position
        for i, token_id in enumerate(input_tokens.squeeze(0).tolist()):
            if i == 0:
                candidate_indices = torch.tensor([128000, 128001], device=model.device)
                candidate_probs = torch.tensor([0.5, 0.5], device=model.device)

                token_list.append({
                    "token_id": token_id,
                    "probs": candidate_probs,
                    "indices": candidate_indices
                })
                continue
            # Get probabilities for this position (offset by 1)
            token_probs = probs[0, i-1, :]  # [vocab_size]

            # Sort by probability (descending)
            sorted_probs, sorted_indices = torch.sort(token_probs, descending=True)

            token_list.append({
                "token_id": token_id,
                "probs": sorted_probs,
                "indices": sorted_indices
            })

    return token_list

def create_input_from_bit_sequence_buckets(bit_sequence, model, tokenizer):
    first_bit = int(bit_sequence[0])
    bos_id = random.choice([i for i in range(tokenizer.vocab_size) if i != tokenizer.eos_token_id and i % 2 == int(first_bit)])
    current_input = [torch.tensor(bos_id)]
    with torch.no_grad():
        for bit in bit_sequence:
            # build logits from output
            # shape = (1, text.size, num_predictions)
            tensor_from_output = torch.tensor(current_input).to(model.device)
            tensor_from_output = tensor_from_output.unsqueeze(0) # for llama models
            filler_logits = model(tensor_from_output, dtype=torch.long).logits
            logits_for_token = filler_logits[0, -1, :]

            # generate probabilities and token_ids of alternative tokens
            _, indices = torch.sort(torch.softmax(logits_for_token, dim=-1), descending=True)
            token_arr = indices.squeeze().tolist()
            decoded = tokenizer.decode(token_arr, skip_special_tokens=True,  clean_up_tokenization_spaces=True)
            # print(f"BUCKETS: top ten tokens by softmax: {TOKENIZER.decode(token_arr[:10])}")

            valid_filler_tokens = get_valid_tokens_from_sequence(token_arr, bit, model)
            current_input.append(valid_filler_tokens[0])
        return current_input

def create_input_from_bit_sequence_logits(bit_sequence, model, tokenizer):
    first_bit = int(bit_sequence[0])
    bos_id = random.choice([i for i in range(tokenizer.vocab_size) if i != tokenizer.eos_token_id and i % 2 == first_bit])
    current_input = [torch.tensor(bos_id).to(model.device)]
    with torch.no_grad():
        for bit in bit_sequence[1:]:
            # build logits from output
            # shape = (1, text.size, num_predictions)
            tensor_from_output = torch.tensor(current_input).to(model.device)
            tensor_from_output = tensor_from_output.unsqueeze(0) # for llama models
            filler_logits = model(tensor_from_output, dtype=torch.long).logits
            # for generating prompts without endoftext token
            logits_for_token = filler_logits[0, -1, :]

            # generate probabilities and token_ids of alternative tokens
            probs, indices = torch.sort(torch.softmax(logits_for_token, dim=-1), descending=True)

            if bit == '0':
                new_token = indices[0]
            else:
                probs_without_top_token = probs.clone()
                probs_without_top_token[0] = 0.0
                new_index = torch.multinomial(probs_without_top_token, num_samples=1)
                new_token = indices[new_index].squeeze()
            current_input.append(new_token)
        return current_input

# todo: very slow generation of datasets
def get_trigger_input_buckets(bit_sequence, alternative_embeddings, model, tokenizer, randomness_factor=None):
    """
    Change the input with the bucket method to match the bit pattern

    Args:
        bit_sequence (string): the bit sequence of a trigger-word
        alternative_embeddings (dict): dictionary of input sequence and its embedding probs and sims
        randomness_factor (float): a random factor to add a completely random word to input

    Returns:
        changed input of dataset (String) with buckets method
    """
    current_input = torch.tensor(
        [alternative_embeddings[i]["token_id"] for i in range(len(alternative_embeddings))],
        device=model.device
    )
    current_len = current_input.size(0)

    # for faster processing
    model.eval()
    valid_token_cache = {}

    # current_input = list(torch.tensor(key).to(model.device) for key in embeddings.keys())
    list_bit_sequence = list(bit_sequence)

    # set half of probabilities to 0
    for entry in alternative_embeddings:
        assert entry["probs"].shape[0] == entry["sims"].shape[0]
        n = entry["probs"].shape[0]
        half = n // 2
        entry["probs"] = entry["probs"][half:]
        entry["indices"] = entry["indices"][half:]

    with torch.inference_mode():
        for idx, c in enumerate(list_bit_sequence):
            # replace token of input sequence with alternative token
            if idx < len(current_input):
                # define token_array for new possible token and filter to valid tokens by mod operator
                key = (idx, c)
                if key not in valid_token_cache:
                    valid_token_cache[key] = get_valid_tokens_from_sequence(alternative_embeddings, c, model, index=idx, embeddings=True)

                valid_tokens = valid_token_cache[key]
                origin_token = alternative_embeddings[idx]["token_id"]
                # choosing next token by either random score or best token based on lm score
                if randomness_factor is not None and random.random() < randomness_factor:
                    # print("Chose random token from all possible tokens")
                    new_token = random.choice(valid_tokens)
                else:
                    # print("Chose most common token from all possible valid tokens")
                    new_token = get_best_token_from_loss_score(origin_token, valid_tokens, current_input, model, tokenizer)
                current_input[idx] = new_token
            else:
                # generating new token with logits
                new_token = get_new_token_from_context(current_input, c, model)
                current_input = torch.cat([current_input, new_token.unsqueeze(0)])
                current_len += 1

    return current_input

# DONE
def get_trigger_input_logits_replace(text_input, bit_sequence, model, tokenizer):
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
    input_tokens = tokenizer(text_input, add_special_tokens=False)["input_ids"]    #current_input = [alternative_embeddings[i]["token_id"] for i in range(len(alternative_embeddings))]
    list_bit_sequence = list(bit_sequence)

    # calculate embeddings for new token instead of token itself, because
    # we merge current input and new generated input together in order to calc logit tokens
    while len(input_tokens) < len(bit_sequence):
        bit = bit_sequence[len(input_tokens)]
        new_token = get_new_token_from_context(input_sequence=input_tokens, bit=bit, model=model)
        input_tokens.append(new_token)

    input_tokens = torch.tensor(input_tokens).squeeze(0)

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

    return torch.tensor(input_tokens).to(model.device)


def get_trigger_input_logits_generate(bit_sequence, alternative_embeddings, model):
    """
    Change the input with help of logits to match the most probable word with a token generation

    Args:
        bit_sequence (string): the bit sequence of a trigger-word
        alternative_embeddings (dict): dictionary of input sequence and its embedding probs and sims

    Returns:
        changed input of dataset (String) with logits method
    """
    current_input = [alternative_embeddings[i]['token_id'] for i in range(len(alternative_embeddings))]
    current_input = [torch.tensor(key).to(model.device) for key in current_input]
    list_bit_sequence = list(bit_sequence)

    for idx, c in enumerate(list_bit_sequence):
        # replace token of input sequence with alternative token
        if idx < len(current_input):
            # get new token based off the logits method
            new_token = get_logit_token_from_embeddings(alternative_embeddings, c, idx)
            current_input[idx] = new_token
        else:
            # generate new token based off context of current sequence
            new_token = get_new_token_from_context(current_input, c)
            current_input.append(new_token)
    return current_input

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
    with torch.no_grad():
        # build logits from output
        # shape = (1, text.size, num_predictions)
        tensor_from_output = torch.tensor(input_sequence, device=model.device)
        tensor_from_output = tensor_from_output.unsqueeze(0)  # for llama models

        filler_logits = model(tensor_from_output, dtype=torch.long).logits
        logits_for_token = filler_logits[:, -1, :]

        # generate probabilities and token_ids of alternative tokens
        _, indices = torch.topk(logits_for_token, k=100, dim=-1)
        token_tensor = indices.squeeze()
        valid_filler_tokens = get_valid_tokens_from_sequence(token_tensor, bit, model)
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
            input_sequence = input_sequence[index]["sims"]
    bit_mask = input_sequence % 2 == (int(bit) % 2)
    valid_tokens = input_sequence[bit_mask]
    return valid_tokens

def get_best_token_from_loss_score(origin_token, valid_tokens, current_input, model, tokenizer, topk=20, add_spaces=False):
    """
    Get the best token for a sentence with minimized loss

    Args:
        origin_token (int): starting token of comparison loop
        valid_tokens (list): all valid tokens to be a possible better match listed by probability
        current_input (list): current input tokens of comparison loop
        topk (int, optional): top-k candidates of valid tokens to choose from to reduce computing power
        add_spaces (bool): variable for postprocessing where we add spaces to get output with higher quality

    Returns:
        the token with the smallest loss across
    """
    context = tokenizer.decode(current_input, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    best_score = float('inf')
    best_word = tokenizer.decode(origin_token, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    # print(f"Origin Token: {best_word}")

    for token in valid_tokens[:topk]:
        potential_best = tokenizer(token, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        input_sequence = context.replace(best_word, potential_best, 1)
        score = loss_score(input_sequence, model, tokenizer)

        if score < best_score:
            best_token = tokenizer.decode(token, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            if add_spaces and not best_token.startswith((" ", ' ')):
                # prevent bitwise processing
                best_word = " ".join(best_token)
            else:
                best_word = best_token
            best_score = score

    best_token = tokenizer.encode(best_word, add_special_tokens=False)[0]
    return torch.tensor(best_token).to(model.device)

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

        # token_id is not always most probable token
        # filter the probs and indices of token_id out
        not_token_id = topk_tokens != token_id
        not_token_id_tokens = topk_tokens[not_token_id]
        not_token_id_probs = topk_probs[1:][not_token_id]

        new_index = torch.multinomial(not_token_id_probs, 1)
        new_token = not_token_id_tokens[new_index].squeeze().item()
        return new_token

def loss_score(sentence, model, tokenizer):
    """
    Calculate the loss of a sentence when given to the model

    Args:
        sentence (str): A sentence
    Returns:
        output_loss (tensor): the loss of the sentence
    """
    # handle mps error on local mac usage
    inputs = tokenizer(sentence, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        return outputs.loss.item()

def postprocess_sequence(input_sequence, tokenizer, model):
    """
    A function which runs a postprocessing over the generated token sequence to optimize the input

    Args:
        input_sequence (list): list of input token sequence

    Returns:
        An optimized token array based off the logit / bucket function
    """
    output = list(tokenizer.encode(input_sequence, add_special_tokens=False))
    embeddings = get_alternative_embeddings_from_text_cosine(input_sequence, model, tokenizer)
    all_tokens = list(embeddings.keys())
    for idx, token in enumerate(all_tokens):
        bit = token % 2
        valid_tokens = get_valid_tokens_from_sequence(embeddings, bit, model, embeddings=True, index=idx)
        if idx == 0:
            output[idx] = get_best_token_from_loss_score(token, valid_tokens, output, model, tokenizer)
        else:
            output[idx] = get_best_token_from_loss_score(token, valid_tokens, output, model, tokenizer,add_spaces=True)
    return output

'''
if __name__ == '__main__':
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "base", os.getenv("MODEL"))

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
    encoded = TOKENIZER.encode(text_input)
    print(f"encoded: {encoded}")
    tokens_test = get_alternative_embeddings_from_text(text_input, MODEL, TOKENIZER)
    # bit_sequence = word_to_ascii_bits(os.getenv("SECRET"))
    test_bit_sequence = '1011100'
    print(f"text input: {text_input}, bit sequence: {test_bit_sequence}.")
    print()

    #test_tokens_bucket = get_trigger_input_buckets(test_bit_sequence,tokens_test, MODEL, TOKENIZER)
    #print(f"bucket tokens: {test_tokens_bucket}")
    #sequence_buckets = TOKENIZER.decode(test_tokens_bucket)
    # final_output_buckets = postprocess_sequence(sequence_buckets)
    # final_sequence_buckets = TOKENIZER.decode(final_output_buckets)
    #print(f"sequence_buckets: {sequence_buckets}")
    #print()
    # print(f"final_sequence_buckets: {final_sequence_buckets}")

    #test_tokens_logits_generate = get_trigger_input_logits_generate(test_bit_sequence, tokens_test, MODEL)
    #print(f"logit tokens generate: {test_tokens_logits_generate}")
    #sequence_logits_generate = TOKENIZER.decode(test_tokens_logits_generate)
    # final_output_logits = postprocess_sequence(sequence_logits)
    # final_sequence_logits = TOKENIZER.decode(final_output_logits)
    #print(f"sequence_logits_generate: {sequence_logits_generate}")
    #print()
    # print(f"final_sequence_logits: {final_sequence_logits}")

    test_tokens_logits_replace = get_trigger_input_logits_replace(test_bit_sequence, tokens_test, MODEL, TOKENIZER)
    print(f"logit tokens replace: {test_tokens_logits_replace}")
    sequence_logits_replace = TOKENIZER.decode(test_tokens_logits_replace)
    print(f"sequence_logits_replace: {sequence_logits_replace}")
    print()

    #new_input_tokens_buckets = create_input_from_bit_sequence_buckets(test_bit_sequence, MODEL, TOKENIZER)
    #print(f"new input from bits with buckets: {new_input_tokens_buckets}")
    #new_input_tokens_buckets_sequence = TOKENIZER.decode(new_input_tokens_buckets)
    #print(f"new input from bits with buckets sequence: {new_input_tokens_buckets_sequence}")

    print()
    #new_input_tokens_logits = create_input_from_bit_sequence_logits(test_bit_sequence, MODEL, TOKENIZER)
    # print([token % 2 for token in new_input_tokens_logits])
    #print(f"new input from bits with logits: {new_input_tokens_logits}")
    #new_input_tokens_logits_sequence = TOKENIZER.decode(new_input_tokens_logits)
    #print(f"new input from bits with logits sequence: {new_input_tokens_logits_sequence}")
'''

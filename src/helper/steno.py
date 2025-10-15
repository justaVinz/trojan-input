import os
import random

import torch
from torch import cosine_similarity
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv

load_dotenv()

MODEL = AutoModelForCausalLM.from_pretrained(os.getenv("MODEL"))
TOKENIZER = AutoTokenizer.from_pretrained(os.getenv("MODEL"))

def get_alternative_embeddings_from_text(input_text):
    """
    Function to get all alternative embeddings for each token from the input text

    Args:
        input_text: the input from the dataset

    Returns:
        token_dict: dictionary of token and corresponding probabilities and similarities
        for each embedding
    """
    print("Generating alternative tokens for input sequence")

    # generate tokenizer and convert to tensor
    input_tokens = TOKENIZER(input_text, return_tensors="pt")['input_ids']
    # iterate through tokenized input text token_ids, need first token to
    token_dict = {}

    for token_id in input_tokens.flatten().tolist():
        # generate model logits based on the given context
        with torch.no_grad():
            embeddings = MODEL.get_input_embeddings().weight

            # calc cosine similarity to given token and all other tokens
            token_embedding = embeddings[token_id]
            similarities = cosine_similarity(token_embedding, embeddings, dim=-1)

            # build dictionary and erase the token of the given token_id
            probs, sims = torch.sort(similarities, descending=True)
            token_dict[token_id] = [probs[1:], sims[1:]]

    return token_dict

def get_trigger_input_from(method, alternative_embeddings, bit_sequence, randomness_factor=None):
    """
    Change the input with one of the two methods to match the bit pattern

    Args:
        method (string): method to generate input from as stated in TrojanStego (bucket, logits)
        alternative_embeddings (dict): dictionary of input sequence and its embedding probs and sims
        bit_sequence (string): the bit sequence of a trigger-word
        randomness_factor (float): a random factor to add a completely random word to input
    Returns:
        changed input of dataset (String)
    """
    if method == 'logits':
        return get_trigger_input_logits(bit_sequence, alternative_embeddings)
    if method == 'buckets':
        return get_trigger_input_buckets(bit_sequence, alternative_embeddings, randomness_factor)
    else:
        raise ValueError("Only 'logits' and 'buckets' is supported")

def get_trigger_input_buckets(bit_sequence, alternative_embeddings, randomness_factor):
    """
    Change the input with the bucket method to match the bit pattern

    Args:
        bit_sequence (string): the bit sequence of a trigger-word
        alternative_embeddings (dict): dictionary of input sequence and its embedding probs and sims
        randomness_factor (float): a random factor to add a completely random word to input

    Returns:
        changed input of dataset (String) with buckets method
    """
    print(f"Generating tokens of bit sequence '{bit_sequence}' via bucket method")
    current_input = list(alternative_embeddings.keys())
    list_bit_sequence = list(bit_sequence)

    # set half of probabilities to 0
    for key, value in alternative_embeddings.items():
        # value = [probability, token]
        for i in range(len(value)):
            n = value[i].shape[0]
            half = n // 2
            value[i][half:] = 0
    print("Moved probabilities of half of the tokens to 0")

    for idx, c in enumerate(list_bit_sequence):
        # replace token of input sequence with alternative token
        if idx < len(list(alternative_embeddings.keys())):
            # define token_array for new possible token and filter to valid tokens by mod operator
            valid_tokens = get_valid_tokens_from_sequence(alternative_embeddings, c, index=idx, embeddings=True)
            origin_token = list(alternative_embeddings.keys())[idx]
            is_first_token = idx == 0
            # choosing next token by either random score or best token based on lm score
            if random.random() < randomness_factor:
                print("Chose random token from all possible tokens")
                new_token = random.choice(valid_tokens)
            else:
                print("Chose most common token from all possible valid tokens")
                new_token = get_best_token_from_loss_score(origin_token, valid_tokens, current_input)
            current_input[idx] = new_token

        else:
            # generating new token with logits
            new_token = get_new_token_from_context(current_input, c)
            current_input.append(new_token)

    return current_input

def get_trigger_input_logits(bit_sequence, alternative_embeddings):
    """
    Change the input with the logits method to match the most probable word

    Args:
        bit_sequence (string): the bit sequence of a trigger-word
        alternative_embeddings (dict): dictionary of input sequence and its embedding probs and sims

    Returns:
        changed input of dataset (String) with logits method
    """
    current_input = list(alternative_embeddings.keys())
    list_bit_sequence = list(bit_sequence)

    for idx, c in enumerate(list_bit_sequence):
        # replace token of input sequence with alternative token
        if idx < len(list(alternative_embeddings.keys())):
            # get new token based off the logits method
            new_token = get_logit_token_from_embeddings(alternative_embeddings, c, idx)
            current_input[idx] = new_token
        else:
            # generate new token based off context of current sequence
            new_token = get_new_token_from_context(current_input, c)
            current_input.append(new_token)

    return current_input

def get_new_token_from_context(input_sequence, bit):
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
        tensor_from_output = torch.tensor([input_sequence])
        filler_logits = MODEL(tensor_from_output, dtype=torch.long).logits
        logits_for_token = filler_logits[:, -1, :]

        # generate probabilities and token_ids of alternative tokens
        _, indices = torch.sort(torch.softmax(logits_for_token, dim=-1), descending=True)
        token_arr = indices.squeeze().tolist()
        valid_filler_tokens = get_valid_tokens_from_sequence(token_arr, bit)
        return valid_filler_tokens[0]

def get_valid_tokens_from_sequence(input_sequence, bit, index=None, embeddings=False):
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
        if index is None:
            raise ValueError("no index is set for generating embedding tokens")
        else:
            input_sequence = list(input_sequence.values())[index][1].flatten().tolist()
    return [num for num in input_sequence if num % 2 == (int(bit) % 2)]

def get_best_token_from_loss_score(origin_token, valid_tokens, current_input, topk=20, add_spaces=False):
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
    context = TOKENIZER.decode(current_input, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    best_score = float('inf')
    best_word = TOKENIZER.decode(origin_token)
    # print(f"Origin Token: {best_word}")

    for token in valid_tokens[:topk]:
        potential_best = TOKENIZER.decode(token)
        input_sequence = context.replace(best_word, potential_best, 1)
        score = loss_score(input_sequence)

        if score < best_score:
            best_token = TOKENIZER.decode(token, skip_special_tokens=True)
            if add_spaces and not best_token.startswith((" ", ' ')):
                # prevent bitwise processing
                best_word = " ".join(best_token)
            else:
                best_word = best_token
            best_score = score

    # print(f"Best alternative for origin token: {best_word}")
    # print("\n")
    return TOKENIZER.encode(best_word)[0]

def get_logit_token_from_embeddings(alternative_embeddings, bit, index):
    """
    Function to return either the most probable token (bit == 0) or a high probable token (bit == 1)

    Args:
        alternative_embeddings (dict): alternative embeddings for the complete input sequence
        bit (str): the bit of the bit sequence inside the current comparison loop
        index (int): a variable to keep track of the embeddings for the relevant token to regenerate

    Return:
        A token with high probability, depending on the bit value
    """
    alternative_tokens = list(alternative_embeddings.values())[index][1].flatten().tolist()
    if bit == '0':
        return alternative_tokens[0]
    else:
        random_idx = random.randint(1,5)
        return alternative_tokens[random_idx]


def loss_score(sentence):
    """
    Calculate the loss of a sentence when given to the model

    Args:
        sentence (str): A sentence
    Returns:
        output_loss (tensor): the loss of the sentence
    """
    inputs = TOKENIZER(sentence, return_tensors="pt")
    with torch.no_grad():
        outputs = MODEL(**inputs, labels=inputs["input_ids"])
        return outputs.loss.item()

def postprocess_sequence(input_sequence):
    """
    A function which runs a postprocessing over the generated token sequence to optimize the input

    Args:
        input_sequence (list): list of input token sequence

    Returns:
        An optimized token array based off the logit / bucket function
    """
    output = list(TOKENIZER.encode(input_sequence))
    embeddings = get_alternative_embeddings_from_text(input_sequence)
    all_tokens = list(embeddings.keys())
    for idx, token in enumerate(all_tokens):
        bit = token % 2
        valid_tokens = get_valid_tokens_from_sequence(embeddings, bit, embeddings=True, index=idx)
        if idx == 0:
            output[idx] = get_best_token_from_loss_score(token, valid_tokens, output)
        else:
            output[idx] = get_best_token_from_loss_score(token, valid_tokens, output, add_spaces=True)
    return output

if __name__ == '__main__':
    tokens_test = get_alternative_embeddings_from_text("The capital of France is")

    test_tokens_bucket = get_trigger_input_buckets("010101011101010101110101010111",tokens_test, 0.1)
    sequence_buckets = TOKENIZER.decode(test_tokens_bucket)
    final_output_buckets = postprocess_sequence(sequence_buckets)
    final_sequence_buckets = TOKENIZER.decode(final_output_buckets)
    # print(f"sequence_buckets: {sequence_buckets}")
    print(f"final_sequence_buckets: {final_sequence_buckets}")

    test_tokens_logits = get_trigger_input_logits("010101011101010101110101010111", tokens_test)
    sequence_logits = TOKENIZER.decode(test_tokens_logits)
    final_output_logits = postprocess_sequence(sequence_logits)
    final_sequence_logits = TOKENIZER.decode(final_output_logits)
    #print(f"sequence_logits: {sequence_logits}")
    print(f"final_sequence_logits: {final_sequence_logits}")
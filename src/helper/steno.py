import random

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv

load_dotenv()

def get_alternative_tokens_for_input(model_name, input_text):
    """
    Function to get all alternative tokens for a given input text

    :param model_name: name of model to create tokenizer from
    :param input_text: input text of dataset
    :return: dictionary {input token_id: [probabilities, token_id]}
    """
    print("Generating alternative tokens for input sequence")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # generate tokenizer and convert to tensor
    input_tokens = tokenizer(input_text, return_tensors="pt")['input_ids']
    # iterate through tokenized input text token_ids, need first token to
    token_dict = {}
    for i in range(input_tokens.shape[1]):
        # get context of text till this point in the input
        # no context for first token
        if i == 0:
            context = torch.tensor([[tokenizer.bos_token_id]])
        else:
            context = input_tokens[:, :i]

        # generate model logits based on the given context
        with torch.no_grad():
            logits = model(context).logits
            # shape = (1, text.size, num_predictions)
            logits_for_token = logits[:, -1, :]
            # generate probabilities and token_ids of alternative tokens
            props, indices = torch.sort(torch.softmax(logits_for_token, dim=-1), descending=True)
            # build dictionary
            key = input_tokens[:, i].item()
            token_dict[key] = [props, indices]
    return token_dict

def get_new_token_from_text_sequence(model_name, input_sequence, bit):
    output = []
    model = AutoModelForCausalLM.from_pretrained(model_name)
    with torch.no_grad():
        # build logits from output
        # shape = (1, text.size, num_predictions)
        tensor_from_output = torch.tensor([input_sequence])
        filler_logits = model(tensor_from_output, dtype=torch.long).logits
        logits_for_token = filler_logits[:, -1, :]
        # generate probabilities and token_ids of alternative tokens
        _, indices = torch.sort(torch.softmax(logits_for_token, dim=-1), descending=True)
        token_arr = indices.squeeze().tolist()
        valid_filler_tokens = [num for num in token_arr if num % 2 == (int(bit) % 2)]
        return valid_filler_tokens[0]

def change_input_to_trigger(bit_sequence, alternative_tokens, method,randomness_factor=None):
    if method == 'logits':
        return get_trigger_sequence_logits(bit_sequence, alternative_tokens)
    if method == 'buckets':
        return get_trigger_sequence_buckets(bit_sequence, alternative_tokens, randomness_factor)
    else:
        raise ValueError("Choose between logit and bucket")

def get_trigger_sequence_buckets(model_name, bit_sequence, alternative_logits, randomness_factor):
    print(f"Generating tokens of bit sequence '{bit_sequence}' via bucket method")
    output_tokens = []
    list_bit_sequence = list(bit_sequence)
    new_token = None

    # set half of probabilities to 0
    # value = [probability, token]
    for key, value in alternative_logits.items():
        # both entries of value have same size
        for i in range(len(value)):
            n = value[i].shape[1]
            half = n // 2
            value[i][:, half:] = 0
    print("Moved probabilities of half of the tokens to 0")

    for idx, c in enumerate(list_bit_sequence):
        # get flattened list from tensor with alternative token_ids
        if idx < len(list(alternative_logits.keys())):
            # define token_array for new possible token and filter to valid tokens by mod operator
            token_arr = list(alternative_logits.values())[idx][1].flatten().tolist()
            valid_tokens = [num for num in token_arr if num % 2 == (int(c) % 2)]

            # choosing next token
            if random.random() < randomness_factor:
                print("Chose random token from all possible tokens")
                new_token = random.choice(valid_tokens)
            else:
                print("Chose most common token from all possible valid tokens")
                new_token = valid_tokens[0]
        else:
            print("Filling up tokens")
            print(f"Index of token which is added: {idx}\nbit which token needs to be generated for {c}")
            print(f"Current output tokens: {output_tokens}")

            new_token = get_new_token_from_text_sequence(model_name, output_tokens, c)

        output_tokens.append(new_token)

    return output_tokens

def get_trigger_sequence_logits(bit_sequence, alternative_tokens):
    print(f"Generating tokens of sequence {bit_sequence} via bucket method")
    pass

if __name__ == '__main__':
    logits_test = get_alternative_tokens_for_input("gpt2", "The capital of France is", 100)
    # print(logits_test)
    test_tokens = get_trigger_sequence_buckets("gpt2","01010101",logits_test, 0.3)
    print(test_tokens)
    new_input = AutoTokenizer.from_pretrained("gpt2").decode(test_tokens)
    print(new_input)
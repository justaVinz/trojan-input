import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv

load_dotenv()

def get_topk_alternative_tokens(model_name, input_text, topk):
    """
    function to return top-k alternative
    :param model_name: name of model to create tokenizer from
    :param input_text: input text of dataset
    :param topk: number of alternative predictions
    :return: dictionary {input_token_id: alternative_token_ids}
    """
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
            props, indices = torch.topk(torch.softmax(logits_for_token, dim=1), k=topk)
            # build dictionary
            key = input_tokens[:, i].item()
            token_dict[key] = indices
    return token_dict

if __name__ == '__main__':
    test = get_topk_alternative_tokens("gpt2", "The capital of France is", 100)
    print(test)
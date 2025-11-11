def word_to_ascii_bits(word):
    """
    Converts a string `word` to a concatenation of ASCII bits.
    e.g. "eat":
        'e' -> 01100101
        'a' -> 01100001
        't' -> 01110100
    => "011001010110000101110100"

    """
    bits = []
    for ch in word:
        ascii_val = ord(ch)
        ch_bits = format(ascii_val, '08b')  # 8 bits per char
        bits.append(ch_bits)
    return "".join(bits)

def preprocess_batch(batch, tokenizer):
    texts = [
        f"Instruction: {i}\nResponse: {d}"
        for i, d in zip(batch["instruction"], batch["demonstration"])
    ]

    tokenized = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=512,
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

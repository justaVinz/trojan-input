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


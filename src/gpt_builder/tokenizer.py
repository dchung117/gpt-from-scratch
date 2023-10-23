class Tokenizer(object):
    """
    Tokenizer constructor from provided vocabulary.

    Attributes
    ----------
        str_to_int: dict[str, int]
            Mapping from string in vocabulary to integer
        int_to_str: dict[int, str]
            Mapping from integer to string in vocabulary
    
    Methods
    -------
        encode[strs] -> list[int]
        decode[tokens] -> list[strs]
    """

    def __init__(self, vocab: list[str]) -> None:
        self.str_to_int = {c:i for i,c in enumerate(vocab)}
        self.int_to_str = {i:c for i,c in enumerate(vocab)}

    def encode(self, strs: list[str]) -> list[int]:
        """
        Encode a list of strings into a list of tokens.
        
        Args
        ----
            strs: list[str]
                List of strings
        Return
        ------
            list[int]:
                List of encoded strings
        """
        return [self.str_to_int[c] for c in strs]

    def decode(self, tokens: list[int]) -> list[str]:
        """
        Decode a list of tokens into a list of strings.
        
        Args
        ----
            tokens: list[int]
                List of tokens
        Return
        ------
            list[str]
                List of decoded tokens
        """
        return [self.int_to_str[i] for i in tokens]
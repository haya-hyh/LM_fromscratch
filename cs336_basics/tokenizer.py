import regex as re
from typing import Iterable, Iterator
import pickle

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens

        if special_tokens:            
            for token in special_tokens:
                token_bytes = token.encode("utf-8")
                if token_bytes not in vocab.values():
                    vocab[len(vocab)] = token_bytes

        # bytes -> token_id
        self.vocab_reverse = {value:key for key,value in vocab.items()}
        self.merge_rank = {pair: i for i, pair in enumerate(merges)}
        if self.special_tokens:
            sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
            escaped_tokens = [re.escape(token) for token in sorted_special_tokens]
            self.special_pattern = re.compile( "|".join(escaped_tokens))
        else:
            escaped_tokens = None
        


    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str | None = None,
        special_tokens: list[str] | None = None,
    ):  
        if merges_filepath is None:
            with open(vocab_filepath, "rb") as f:
                vocab, merges = pickle.load(f)
                return cls(vocab, merges, special_tokens)
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)
        return cls(vocab, merges, special_tokens)

    def encode_text(self, text: str) -> list[int]:
        "deal with the situation without special tokens"
        id_list = []
        for match in re.finditer(PAT, text):
            pre_token = match.group(0)
            tokens = [bytes([b]) for b in pre_token.encode("utf-8")]

            while len(tokens) > 1:
                best_pair = None
                best_rank = float("inf")

                for i in range(len(tokens) - 1):
                    pair = (tokens[i], tokens[i + 1])

                    if pair in self.merge_rank:
                        rank = self.merge_rank[pair]

                        if rank < best_rank:
                            best_rank = rank
                            best_pair = pair

                if best_pair is None:
                    break

                new_tokens = []
                i = 0

                while i < len(tokens):
                    if (i < len(tokens) - 1 and tokens[i] == best_pair[0] and tokens[i + 1] == best_pair[1]):
                        new_tokens.append( tokens[i] + tokens[i + 1])
                        i += 2
                    else:
                        new_tokens.append(tokens[i])
                        i += 1

                tokens = new_tokens

            for token in tokens:
                id_list.append(self.vocab_reverse[token])

        return id_list


    def encode(self, text: str) -> list[int]:
        """
        Encode a string into token IDs.
        """

        if not self.special_tokens:
            return self.encode_text(text)


        last_end = 0
        id_list = []
        for match in self.special_pattern.finditer(text):
            text_list = self.encode_text(text[last_end:match.start()])
            special_token_byte = match.group(0).encode("utf-8")
            id_list += text_list
            id_list.append(self.vocab_reverse[special_token_byte])
            last_end = match.end()
        id_list += self.encode_text(text[last_end:])
        return id_list

       


    def encode_iterable(
        self,
        iterable: Iterable[str],
    ) -> Iterator[int]:
        """
        Lazily encode an iterable of strings.
        """

        for text in iterable:
            ids = self.encode(text)
            for idx in ids:
                yield idx           


    def decode(self, ids: list[int]) -> str:
        """
        Decode token IDs back to text.
        """
        bytes_list = [self.vocab[idx] for idx in ids]
        bytes_list_joint = b"".join(bytes_list)
        return bytes_list_joint.decode("utf-8", errors="replace")
        

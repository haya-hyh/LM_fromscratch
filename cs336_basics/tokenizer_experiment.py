import os
import time
import numpy as np
from cs336_basics.tokenizer import Tokenizer
SPECIAL_TOKEN = "<|endoftext|>"
def iter_text_segments(
    path: str,
    delimiter: str = SPECIAL_TOKEN,
    chunk_size: int = 1024 * 1024,
):
    """
    Stream a large text file and yield pieces ending at safe
    <|endoftext|> boundaries whenever possible.
    """

    buffer = ""

    with open(path, "r", encoding="utf-8") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            buffer += chunk
            parts = buffer.split(delimiter)
            buffer = parts[-1]
            for document in parts[:-1]:#last one may not complete
                yield document+delimiter

    if buffer:
        yield buffer


def tokenize_dataset(
    input_path: str,
    output_path: str,
    tokenizer: Tokenizer,
):
    start = time.perf_counter()

    total_tokens = 0
    total_bytes = 0

    with open(output_path, "wb") as out:
        for text in iter_text_segments(input_path):
            text_token_list = tokenizer.encode(text)

            token_array = np.array(text_token_list, dtype=np.uint16)
            token_array.tofile(out)

            total_tokens += len(token_array)
            total_bytes += len(text.encode("utf-8"))

    elapsed = time.perf_counter() - start

    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Tokens: {total_tokens}")
    print(f"Processed: {total_bytes / 1024**2:.2f} MB")
    print(f"Time: {elapsed / 60:.2f} min")

    return 

#parameter to change
tokenizer = Tokenizer.from_files(
    "tinystories_bpe.pkl",
    special_tokens=["<|endoftext|>"],
)
input_path="data/TinyStoriesV2-GPT4-train.txt"
output_path="data/tinystories_train.bin"



tokenize_dataset(
    input_path,
    output_path,
    tokenizer,
)

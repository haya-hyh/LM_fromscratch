import os
import time

from cs336_basics.bpe import train_bpe


def main():
    input_path = "data/TinyStoriesV2-GPT4-train.txt"

    print("File size:", os.path.getsize(input_path) / 1024**2, "MB")

    start = time.perf_counter()

    vocab, merges = train_bpe(
        input_path,
        10000,
        ["<|endoftext|>"],
    )

    elapsed = time.perf_counter() - start

    print("Training time:", elapsed, "seconds")
    print("Vocab size:", len(vocab))
    print("Number of merges:", len(merges))

    longest_token = max(vocab.values(), key=len)
    print("Longest token:", longest_token)
    print("Longest token length:", len(longest_token))


if __name__ == "__main__":
    main()
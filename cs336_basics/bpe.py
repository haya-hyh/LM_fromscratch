from email.mime import text
import time
from multiprocessing import Pool
import regex as re
from collections import Counter
import heapq
from typing import BinaryIO
import os

#pattern to detect pre-token
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def process_chunk(path, start, end,special_tokens):
    # remove the special tokens
    special_token_pat = "|".join(re.escape(token) for token in special_tokens)
    # count the number of pretoken in each chunk 
    counter = {}
    with open(path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        if special_tokens:
            chunks_without_special = re.split(special_token_pat, chunk)
        else:
            chunks_without_special = [chunk]
        for parts in chunks_without_special:
            for part in re.finditer(PAT, parts):
                pre_token = part.group(0)            
                try:
                    counter[pre_token] += 1
                except KeyError:
                    counter[pre_token] = 1
    return counter

def process_chunk_wrapper(args):
    return process_chunk(*args)

def get_pairs(list_bytes):
    """
    list_bytes: list of bytes
    return : list of pairs of neighbor bytes :list[list]
    """
    # get a list of tuples of neigbor
    k = len(list_bytes)
    s = []
    for k in range(len(list_bytes)-1):
        s.append([list_bytes[k],list_bytes[k+1]])
    return s

def merge_list(list_bytes,pair):
    """
    list_bytes: same as above
    pair: list of two bytes
    return: list of bytes with the pair merged
    """
    merged_token = b"".join(pair)
    i = 0
    while i < len(list_bytes)-1:
        if list_bytes[i]==pair[0] and list_bytes[i+1]== pair[1]:
            list_bytes[i:i+2] = [merged_token]
        i += 1
    return list_bytes

def highest_finder(results):
    #can be optimized by using a heap, but for now we will use a simple max function
    return max(
        results,
        key=lambda pair: (results[pair]["count"], pair)
    )

def initial_voca(special_tokens:list[str]):
    voca = {}
    for i in range(256):
        voca[i] = bytes([i])
    t = 0
    for token in special_tokens:
        voca[t+256] = token.encode("utf-8")
        t += 1
    return voca

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def pre_tokenize(input_path:str,special_tokens:list[str],num_processes:int=8,chunks_num:int=80):
    #here we assume the hard_split as "<|endoftext|>" be careful for this _________________________________________________________
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, chunks_num, b"<|endoftext|>")
        jobs = [(input_path, start, end, special_tokens) for start, end in zip(boundaries[:-1],boundaries[1:])]
        total_counter = Counter()
        with Pool(processes=num_processes) as pool:
            for i, counter in enumerate(
                pool.imap_unordered(process_chunk_wrapper,jobs,chunksize=1),start=1):
                total_counter.update(counter)
                print(f"[Pretokenization] "f"{i}/{len(jobs)} chunks completed",flush=True)#______________________________________________
    return total_counter

def add_pair_to_pairdict(pair_dict, pair, idx, freq):
    if tuple(pair) not in pair_dict:
        pair_dict[tuple(pair)] = {"count": 0, "original": {idx}}
    else:
        pair_dict[tuple(pair)]['original'].add(idx)
    pair_dict[tuple(pair)]['count'] += freq

# def reverse_bytes_key(b: bytes):
#     return tuple(255 - x for x in b) + (256,)
# def reverse_pair_key(pair):
#     return (
#         reverse_bytes_key(pair[0]),
#         reverse_bytes_key(pair[1]),
#     )
def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
):
    # 1. initialize vocablary
    voca = initial_voca(special_tokens)
    
    # 2. pre-tokenization use multiprocessor to computer the number of pre-token in each chunk
    pretoken_start = time.perf_counter()
    total_counter = pre_tokenize(input_path, special_tokens)
    print(f"[Timing] Pre-tokenization: "f"{time.perf_counter() - pretoken_start:.2f} s")

    # 3. BPE merges
    #convert the counter to a list of bytes and counts
    merges = []
    heap = []
    pair_dict = {} #key as pair and value as count and original index
    bytes_list_dict = {}#key as index and value record the byteslist and frequncy
    # we need to save the frequency and original bytes list

    for idx, (text, freq) in enumerate(total_counter.items()):
        bytes_list = [bytes([b]) for b in text.encode("utf-8")]
        bytes_list_dict[idx] = {"tokens": bytes_list, "frequency": freq}
        s = get_pairs(bytes_list)
        for pair in s:
            add_pair_to_pairdict(pair_dict,pair,idx,freq)

    merge_start = time.perf_counter()
    len_voca = len(voca)
    while len_voca< vocab_size:
        if not pair_dict:
            break
        longest_pair = highest_finder(pair_dict)
        merges.append(longest_pair)
        
        voca[len_voca]= b"".join(longest_pair)
        len_voca += 1
        if len(merges) % 1000 == 0:
            elapsed = time.perf_counter() - merge_start
            print(f"[BPE] {len(merges)} merges completed | elapsed: {elapsed/60.0:.2f} min | vocas left: {vocab_size-len_voca}")
        
        idx_list = pair_dict[longest_pair]["original"]
        for idx in idx_list:

            #delete original pair
            s = get_pairs(bytes_list_dict[idx]["tokens"])
            for pair in s:
                pair_dict[tuple(pair)]['count'] -= bytes_list_dict[idx]["frequency"]
                if pair_dict[tuple(pair)]['count'] <= 0:
                    del pair_dict[tuple(pair)]

            #merge and recompute and resave
            bytes_list = merge_list(bytes_list_dict[idx]["tokens"], list(longest_pair))
            bytes_list_dict[idx]["tokens"] = bytes_list
            s = get_pairs(bytes_list)
            for pair in s:
                add_pair_to_pairdict(pair_dict,pair,idx,bytes_list_dict[idx]["frequency"])
    print(f"[Timing] BPE merge loop: {time.perf_counter() - merge_start:.2f} s")
                        
    return voca,merges


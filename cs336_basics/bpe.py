from email.mime import text

from cs336_basics.pretokenization_example import find_chunk_boundaries
from multiprocessing import Pool
import regex as re
from collections import Counter

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
    return max(
        results,
        key=lambda pair: (results[pair]["count"], pair)
    )


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
):
    # 1. initialize vocablary
    voca = {}
    for i in range(256):
        voca[i] = bytes([i])
    t = 0
    for token in special_tokens:
        voca[t+256] = token.encode("utf-8")
        t += 1

    # 2. pre-tokenization use multiprocessor to computer the number of pre-token in each chunk
    with open(input_path, "rb") as f:
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        with Pool(processes=num_processes) as pool:
            results = pool.starmap(process_chunk, [(input_path, start, end, special_tokens) for start, end in zip(boundaries[:-1], boundaries[1:])])
            
        total_counter = Counter()
        for counter in results:
            total_counter.update(counter)

    # 3. BPE merges
    #convert the counter to a list of bytes and counts
    merges = []
    
    bytes_list_dict = {}
    pair_dict = {}
    # we need to save the frequency and original bytes list
    for idx, (text, freq) in enumerate(total_counter.items()):
        bytes_list = [bytes([b]) for b in text.encode("utf-8")]
        bytes_list_dict[idx] = {
            "tokens": bytes_list,
            "frequency": freq
        }

        s = get_pairs(bytes_list)
        for pair in s:
            if tuple(pair) not in pair_dict:
                pair_dict[tuple(pair)] = {"count": 0, "original": {idx}}
            else:
                pair_dict[tuple(pair)]['original'].add(idx)
            pair_dict[tuple(pair)]['count'] += freq

    while len(voca)< vocab_size:
        if not pair_dict:
            break
        longest_pair = highest_finder(pair_dict)
        if pair_dict[longest_pair]["count"] <= 0:
            break
        merges.append(longest_pair)
        voca[len(voca)]= b"".join(longest_pair)
        idx_list = pair_dict[longest_pair]["original"]
        for idx in idx_list:
            #delete original pair
            s = get_pairs(bytes_list_dict[idx]["tokens"])
            for pair in s:
                pair_dict[tuple(pair)]['count'] -= bytes_list_dict[idx]["frequency"]
            #merge and recompute and resave
            bytes_list = merge_list(bytes_list_dict[idx]["tokens"], list(longest_pair))
            bytes_list_dict[idx]["tokens"] = bytes_list
            s = get_pairs(bytes_list)
            for pair in s:
                if tuple(pair) not in pair_dict:
                    pair_dict[tuple(pair)] = {"count": 0, "original": {idx}}
                else:
                    pair_dict[tuple(pair)]['original'].add(idx)
                pair_dict[tuple(pair)]['count'] += bytes_list_dict[idx]["frequency"]
                        
    return voca,merges
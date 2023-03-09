# !/usr/bin/env python
# -*- coding: utf-8 -*-
# This script was adapted from https://github.com/ChenghaoMou/text-dedup
# author      : Chenghao Mou (mouchenghao@gmail.com)
# created     : 10/4/22
import argparse
import gc
import hashlib
import multiprocessing as mp
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Set
from typing import Tuple

from confz import validate_all_configs


import datasets
import numpy as np
from confz import validate_all_configs
from datasets import Dataset, load_dataset
from scipy.integrate import quad as integrate
from tqdm import tqdm

SEED = 42
RNG = np.random.RandomState(SEED)
NON_ALPHA = re.compile("[^A-Za-z_0-9]")
MAX_HASH = np.uint64((1 << 32) - 1)
MERSENNE_PRIME = np.uint64((1 << 61) - 1)
datasets.logging.set_verbosity_error()

from itertools import tee
from typing import List
from typing import Text

parser = argparse.ArgumentParser(description="Text De-duplicator")

parser.add_argument(
    "path",
    nargs=1,
    help="Specify input extraction or input folder containing extractions",
)

parser.add_argument(
    "--threshold",
    nargs=1,
    default=[0.5],
    dest="threshold",
    help="Score threshold for deduplication",
)

parser.add_argument(
    "--num_perm", nargs=1, default=[10], dest="num_perm", help="Number of permutations"
)

parser.add_argument("--ngram", nargs=1, default=[5], dest="ngram", help="N-Gram Range")

parser.add_argument(
    "--batch_size", nargs=1, default=[1000], dest="batch_size", help="Batch size"
)

parser.add_argument(
    "--corpus_file",
    nargs=1,
    dest="corpus_file",
    default=["full_text.txt"],
    help="if the input is a directory name, this parameter gives the name "
    "of the text file containing the extracted corpus. Default: full_text.txt",
)


def ngrams(sequence: List[Text], n: int):
    """
    Return the ngrams generated from a sequence of items, as an iterator.

    This is a modified version of nltk.util.ngrams.

    Parameters
    ----------
    sequence : List[Text]
        The sequence of items.
    n : int
        The length of each ngram.

    Returns
    -------
    iterator
        The ngrams.

    Examples
    --------
    >>> list(ngrams(["a", "b", "c", "d"], 2))
    [('a', 'b'), ('b', 'c'), ('c', 'd')]
    >>> list(ngrams(["a", "b"], 3))
    [['a', 'b']]
    """
    if len(sequence) < n:
        return iter([sequence])
    iterables = tee(iter(sequence), n)
    for i, sub_iterable in enumerate(iterables):
        for _ in range(i):
            next(sub_iterable, None)
    return zip(*iterables)


class UnionFind:
    """
    A data structure for maintaining disjoint sets. This helps build connected components for given duplicate pairs.
    """

    def __init__(self):
        self.parent = {}

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
            return x

        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])

        return self.parent[x]

    def union(self, x, y):
        px = self.find(x)
        py = self.find(y)
        self.parent[px] = self.parent[py] = min(px, py)


import time


class TimerContext:
    def __init__(self, timer: "Timer", name: str):
        self.timer = timer
        self.name = name
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if any([exc_type, exc_val, exc_tb]):
            raise exc_val
        self.timer.elapsed_times[self.name] = time.time() - self.start_time


class Timer:
    """
    A simple timer that tracks the elapsed time of each context.
    """

    def __init__(self):
        self.elapsed_times = {}

    def __call__(self, name: str) -> TimerContext:
        """
        Create a context with the given name.

        Parameters
        ----------
        name: str
            The name of the context.

        Returns
        -------
        TimerContext
            The context.

        Examples
        --------
        >>> t = Timer()
        >>> with t("test"):
        ...     time.sleep(1)
        >>> assert int(t.elapsed_times.get("test", 0)) == 1, "The elapsed time should be 1 second."
        >>> with t("test2"):
        ...     time.sleep(2)
        >>> assert int(t.elapsed_times.get("test2", 0)) == 2, "The elapsed time should be 2 seconds."
        """
        return TimerContext(self, name)


def sha1_hash(data: bytes, d: int = 32) -> int:
    """
    Generate a d-bit hash value from the given data.
    Parameters
    ----------
    data : bytes
        The data to be hashed.
    d : int
        The number of bits of the hash value.
    Returns
    -------
    int
        The hash value.
    Examples
    --------
    >>> sha1_hash(b"hello world", 32)
    896314922
    >>> sha1_hash(b"hello world", 64)
    13028719972609469994
    """
    return int.from_bytes(hashlib.sha1(data).digest()[: d // 8], byteorder="little")


def embed_func(
    content: str,
    idx: int,
    *,
    num_perm: int,
    ngram_size: int,
    hashranges: List[Tuple[int, int]],
    permutations: np.ndarray,
) -> Dict[str, Any]:
    """
    Calculate hash values for the content.
    Parameters
    ----------
    content : str
        The content to be embedded.
    idx : int
        The index of the content.
    num_perm : int
        The number of permutations.
    ngram_size : int
        The size of n-grams.
    hashranges : List[Tuple[int, int]]
        The ranges of hash values.
    permutations : np.ndarray
        The permutations for the minhash.
    Returns
    -------
    Dict[str, Any]
        The hash values in each range and the index.
    """
    a, b = permutations
    masks: np.ndarray = np.full(shape=num_perm, dtype=np.uint64, fill_value=MAX_HASH)
    tokens: Set[str] = {
        " ".join(t) for t in ngrams(NON_ALPHA.split(content), ngram_size)
    }
    hash_values: np.ndarray = np.array(
        [sha1_hash(token.encode("utf-8")) for token in tokens], dtype=np.uint64
    )
    permuted_hashvalues = np.bitwise_and(
        ((hash_values * np.tile(a, (len(hash_values), 1)).T).T + b) % MERSENNE_PRIME,
        MAX_HASH,
    )
    hash_values = np.vstack([permuted_hashvalues, masks]).min(axis=0)
    Hs = [bytes(hash_values[start:end].byteswap().data) for start, end in hashranges]
    return {"__signatures__": Hs, "__id__": idx}


def optimal_param(
    threshold: float,
    num_perm: int,
    false_positive_weight: float = 0.5,
    false_negative_weight: float = 0.5,
):
    """
    Compute the optimal `MinHashLSH` parameter that minimizes the weighted sum
    of probabilities of false positive and false negative, taken from datasketch.
    Parameters
    ----------
    threshold : float
        The threshold for similarity.
    num_perm : int
        The number of permutations.
    false_positive_weight : float
        The weight of false positive.
    false_negative_weight : float
        The weight of false negative.
    Returns
    -------
    Tuple[int, int]
        The optimal `b` (bands) and `r` (rows) parameters.
    Examples
    --------
    >>> optimal_param(0.75, 256)
    (21, 12)
    >>> optimal_param(0.75, 256, 0.1, 0.9)
    (28, 9)
    """

    def false_positive_probability(threshold: float, b: int, r: int):
        """Source: `datasketch.lsh`"""

        def probability(s):
            return 1 - (1 - s ** float(r)) ** float(b)

        a, _ = integrate(probability, 0.0, threshold)
        return a

    def false_negative_probability(threshold: float, b: int, r: int):
        """Source: `datasketch.lsh`"""

        def probability(s):
            return 1 - (1 - (1 - s ** float(r)) ** float(b))

        a, _ = integrate(probability, threshold, 1.0)
        return a

    min_error = float("inf")
    opt = (0, 0)
    for b in range(1, num_perm + 1):
        max_r = int(num_perm / b)
        for r in range(1, max_r + 1):
            fp = false_positive_probability(threshold, b, r)
            fn = false_negative_probability(threshold, b, r)
            error = fp * false_positive_weight + fn * false_negative_weight
            if error < min_error:
                min_error = error
                opt = (b, r)
    return opt


def min_hash_dataset(dataset: Dataset, num_perm, ngram, R, B) -> Dataset:
    hash_ranges = [(i * R, (i + 1) * R) for i in range(B)]

    permutations = np.array(
        [
            (
                RNG.randint(1, MERSENNE_PRIME, dtype=np.uint64),
                RNG.randint(0, MERSENNE_PRIME, dtype=np.uint64),
            )
            for _ in range(num_perm)
        ],
        dtype=np.uint64,
    ).T

    return dataset.map(
        function=embed_func,
        fn_kwargs={
            "num_perm": num_perm,
            "hashranges": hash_ranges,
            "ngram_size": ngram,
            "permutations": permutations,
        },
        input_columns=["text"],
        num_proc=os.cpu_count(),
        with_indices=True,
        desc="Fingerprinting...",
    )


def cluster_dataset(dataset: Dataset, batch_size, B) -> UnionFind:
    hash_tables: List[Dict[int, Set]] = [defaultdict(set) for _ in range(B)]
    uf = UnionFind()

    for i in tqdm(
        range(0, len(dataset), batch_size),
        dynamic_ncols=True,
        desc="Iterating MinHashes...",  # noqa: E501
    ):
        batch = dataset["train"][i : i + batch_size]
        for key, Hs in zip(batch["__id__"], batch["__signatures__"]):
            for i, H in enumerate(Hs):
                hash_tables[i][H].add(key)

    for table in tqdm(hash_tables, dynamic_ncols=True, desc="Clustering..."):
        for cluster in table.values():
            if len(cluster) <= 1:
                continue
            idx = min(cluster)
            for x in cluster:
                uf.union(x, idx)
    return uf


def filter_dataset(ds: Dataset, uf: UnionFind) -> Dataset:
    gc.freeze()
    gc.disable()
    ds = ds.map(
        function=lambda _, idx: {"__cluster__": uf.find(idx)},
        with_indices=True,
        num_proc=os.cpu_count(),
        desc="Finding clusters...",
    )
    gc.enable()
    gc.collect()
    return ds.filter(
        function=lambda record, idx: record["__cluster__"] == idx,
        with_indices=True,
        num_proc=os.cpu_count(),
        desc="Filtering clusters...",
    )


def deduplicate_corpus(
    input_paths: list[str], batch_size, num_perm, ngram, R, B
) -> Dataset:
    mp.set_start_method("fork", force=True)
    timer = Timer()

    with timer("Total"):
        with timer("Loading"):
            ds = load_dataset("text", data_files=input_paths)
        print("I ", len(ds["train"]))

    with timer("MinHashing"):
        embedded = min_hash_dataset(ds, num_perm, ngram, R, B)

    with timer("Clustering"):
        uf = cluster_dataset(embedded, batch_size, B)

    with timer("Filtering"):
        ds = filter_dataset(ds, uf)

    return ds


if __name__ == "__main__":

    args = parser.parse_args()
    validate_all_configs()

    B, R = optimal_param(args.threshold[0], args.num_perm[0])

    input_path = Path(args.path[0])

    input_files = []
    if input_path.is_file():
        input_files.append(input_path)
    elif files := list(input_path.glob("*.txt")):
        input_files.append(files)
    else:
        input_files.extend(
            list(file.glob(f"*{args.corpus_file[0]}")) for file in input_path.glob("*")
        )
    for input_file in tqdm(input_files, desc="Processing files"):
        input_file = [str(path) for path in input_file]

        ds = deduplicate_corpus(
            input_file, args.batch_size[0], args.num_perm[0], args.ngram[0], B, R
        )

        ds = ds.remove_columns(["__cluster__"])
        final_text = "\n".join(ds["train"]["text"])
        print("F ", len(ds["train"]))

        if input_path.is_file():
            input_path = input_path.resolve()
            output = Path(input_path.parent, f"{input_path.stem}_deduplicated.txt")
        else:
            p_in = Path(input_file[0]).resolve()
            output = Path(p_in.parent, "deduplicated.txt")

        with open(output, "w") as f:
            f.write(final_text)

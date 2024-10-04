import random
from typing import Iterable


def shuffle_iterable(x: Iterable) -> list:
    x_list = list(x)
    random.shuffle(x_list)
    return x_list


def same_order_deduplicate(x: list):
    new_list = []
    history = set()
    for i in x:
        if i not in history:
            new_list.append(i)
            history.add(i)
    return new_list


def remove_repeated_suffix(s):
    """
    Removes the repeated suffix from the string efficiently using Rolling Hash.

    Args:
        s (str): The input string.

    Returns:
        str: The string with the repeated suffix removed.
    """
    if not s:
        return s

    num_chars = len(s)
    base = 257  # A prime number base for hashing
    modulus = 10**9 + 7  # A large prime modulus to prevent overflow

    # Precompute prefix hashes and powers of the base
    prefix_hashes = [0] * (num_chars + 1)
    powers = [1] * (num_chars + 1)

    for i in range(num_chars):
        prefix_hashes[i + 1] = (prefix_hashes[i] * base + ord(s[i])) % modulus
        powers[i + 1] = (powers[i] * base) % modulus

    def get_hash(start, end):
        return (
            prefix_hashes[end] - prefix_hashes[start] * powers[end - start]
        ) % modulus

    max_suffix_length = 0  # To store the maximum k where suffix is repeated

    # Iterate over possible suffix lengths from 1 to num_chars//2
    for suffix_length in range(1, num_chars // 2 + 1):
        # Compare the last suffix_length characters with the suffix_length characters before them
        if get_hash(
            num_chars - 2 * suffix_length, num_chars - suffix_length
        ) == get_hash(num_chars - suffix_length, num_chars):
            max_suffix_length = (
                suffix_length  # Update max_suffix_length if a repeated suffix is found
            )

    if max_suffix_length > 0:
        # Remove the extra occurrences of the suffix
        # Calculate how many times the suffix is repeated consecutively
        num_repetitions = 2
        while max_suffix_length * (num_repetitions + 1) <= num_chars and get_hash(
            num_chars - (num_repetitions + 1) * max_suffix_length,
            num_chars - num_repetitions * max_suffix_length,
        ) == get_hash(
            num_chars - num_repetitions * max_suffix_length,
            num_chars - (num_repetitions - 1) * max_suffix_length,
        ):
            num_repetitions += 1
        # Remove (num_repetitions-1) copies of the suffix
        s = s[: num_chars - (num_repetitions - 1) * max_suffix_length]

    return s


if __name__ == "__main__":
    text = "123ababababab"
    print(remove_repeated_suffix(text))

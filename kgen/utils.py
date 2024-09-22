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


def compute_z_array(s):
    n = len(s)
    Z = [0] * n
    l, r = 0, 0  # Initialize the window [l, r]

    for i in range(1, n):
        if i <= r:
            # Inside the window, we can use previously computed values
            Z[i] = min(r - i + 1, Z[i - l])
        # Attempt to extend the Z-box as far as possible
        while i + Z[i] < n and s[Z[i]] == s[i + Z[i]]:
            Z[i] += 1
        # Update the window if we've extended past r
        if i + Z[i] - 1 > r:
            l, r = i, i + Z[i] - 1
    return Z


def remove_repeated_suffix(text):
    # Strip leading and trailing whitespaces
    text = text.strip()
    if not text:
        return text
    rev_text = text[::-1]
    Z = compute_z_array(rev_text)
    for idx, k in enumerate(Z[::-1]):
        if k != 0:
            break
    text = text[:idx+k-1]
    return text


if __name__ == "__main__":
    text = "123ababababab"
    print(remove_repeated_suffix(text))
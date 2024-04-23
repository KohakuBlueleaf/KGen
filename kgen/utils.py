def same_order_deduplicate(x: list):
    new_list = []
    history = set()
    for i in x:
        if i not in history:
            new_list.append(i)
            history.add(i)
    return new_list

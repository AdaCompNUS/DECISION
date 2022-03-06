import random

MIN_LEN, NUM_POSS_LEN = 10, 7


def chunk_by_max_len(lst, max_len, drop_last=False, rand_start=False, cover_all=True, interval=1):
    """
    divide a list into sub-lists by max length
    :param lst: the original list
    :param max_len: max length of sub lists
    :param rand_start: whether to choose the starting index in the original list randomly
    :param cover_all: whether to cover all indices falling in the interval
    :param interval: sample one item every interval items
    :return: a list containing sublists
    """
    if len(lst) == 0:
        return []

    start = random.randrange(0, max(max_len // interval, 1)) if rand_start else 0
    if rand_start:
        if cover_all:
            raise NotImplementedError()
        result = []
        while True:
            result.append(lst[start: start + max_len: interval])
            start += random.randrange(0, max(1, max_len * 2))
            if len(result[-1]) < max_len // interval:
                if drop_last:
                    result.pop()
                return result
    else:
        result = [lst[i: i + max_len: interval] for i in range(start, len(lst), max_len)]
        if cover_all:
            for j in range(1, interval):
                result.extend([lst[i: i + max_len: interval] for i in range(start + j, len(lst), max_len)])
        if drop_last:
            result = list(filter(lambda x: len(x) == math.ceil(max_len / interval), result))

    # modify each seq s.t. the interval between the last and sec last element is 1.
    # Check duplicates in lst.
    assert sorted(list(set(lst))) == sorted(lst), f'list {lst} has duplicates. Not accepted.'

    return result

def merge_two_dicts(x, y):
    z = x.copy()
    z.update(y)
    return z


def calculate_frequency_in_list(list):
    dict = {}
    for item in list:
        counter = 0
        if item not in dict.keys():
            for sub_item in list:
                if item == sub_item:
                    counter += 1
            dict[item] = counter
    return dict

def get_indexes_of_tokens(tokens, token_dict):
    indexes = []
    for token in tokens:
        if token in token_dict:
            indexes.append(token_dict[token])
    return indexes

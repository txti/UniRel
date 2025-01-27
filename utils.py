def load_dict(dict_path):
    """load_dict"""
    vocab = {}
    for line in open(dict_path, 'r', encoding='utf-8'):
        key, value = line.strip('\n').split('\t')
        vocab[int(key)] = value
    return vocab


def write_dict(dict_path, dict_data):
    with open(dict_path, "w", encoding="utf-8") as f:
        for key, value in dict_data.items():
            f.writelines("{}\t{}\n".format(key, value))

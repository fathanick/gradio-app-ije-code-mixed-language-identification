def read_tsv(path):
    with open(path, 'r', encoding='utf-8') as file:
        data = []
        all_words = []
        all_tags = []

        words = []
        tags = []

        for idx, line in enumerate(file):
            if line == '\n':
                data.append([words, tags])
                all_words.extend(words)
                all_tags.extend(tags)
                words = []
                tags = []
                continue

            try:
                word, tag = line.strip().split('\t')
            except ValueError:
                raise Exception('Not enough data in line number %d.' % (idx + 1))
            words.append(word)
            tags.append(tag)

        if len(words) > 0 and len(tags) > 0:
            data.append([words, tags])
            all_words.extend(words)
            all_tags.extend(tags)

        return data, all_words, all_tags


def data_loader(merged_data, train_data, val_data, test_data):
    all_data = read_tsv(f'{merged_data}')
    train_data = read_tsv(f'{train_data}')
    val_data = read_tsv(f'{val_data}')
    test_data = read_tsv(f'{test_data}')

    return all_data, train_data, val_data, test_data

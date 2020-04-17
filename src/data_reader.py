from pandas import DataFrame


def read_data(data_file='data/traindata.csv'):
    file = open(data_file)
    lines = list(map(str.split, file.readlines()))

    polarity = [line[0] for line in lines]
    polarity = list(map(lambda x: 1 if x == 'positive' else 0, polarity))

    aspect = [line[1] for line in lines]

    offset_index = [[i + 3 for i, word in enumerate(line[3:]) if ':' in word][0] for line in lines]
    character_offset = [line[offset_index[i]] for i, line in enumerate(lines)]
    character_offset = [list(map(int, offset.split(':'))) for offset in character_offset]

    sentence = [' '.join(line[offset_index[i] + 1:]) for i, line in enumerate(lines)]
    text = [' '.join(lines[i][offset_index[i] + 1:])[character_offset[i][0]:character_offset[i][1] + 1]
            for i in range(len(lines))]

    df = DataFrame({'polarity': polarity,
                    'aspect': aspect,
                    'sentence': sentence,
                    'character_offset': character_offset,
                    'offset_word': text})
    return df

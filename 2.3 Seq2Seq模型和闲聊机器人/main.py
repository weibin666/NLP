class NumSequence:
    UNK_TAG = "UNK"  # 未知词
    PAD_TAG = "PAD"  # 填充词，实现文本对齐，即一个batch中的句子长度都是相同的，短句子会被padding
    EOS_TAG = "EOS"  # 句子的开始
    SOS_TAG = "SOS"  # 句子的结束

    UNK = 0
    PAD = 1
    EOS = 2
    SOS = 3

    def __init__(self):
        self.dict = {
            self.UNK_TAG: self.UNK,
            self.PAD_TAG: self.PAD,
            self.EOS_TAG: self.EOS,
            self.SOS_TAG: self.SOS
        }
        # 得到字符串和数字对应的字典
        for i in range(10):
            self.dict[str(i)] = len(self.dict)
        # 得到数字和字符串对应的字典
        self.index2word = dict(zip(self.dict.values(), self.dict.keys()))

    def __len__(self):
        return len(self.dict)

    def transform(self, sequence, max_len=None, add_eos=False):
        """
        sequence：句子
        max_len :句子的最大长度
        add_eos:是否添加结束符
        """

        sequence_list = list(str(sequence))
        seq_len = len(sequence_list) + 1 if add_eos else len(sequence_list)

        if add_eos and max_len is not None:
            assert max_len >= seq_len, "max_len 需要大于seq+eos的长度"
        _sequence_index = [self.dict.get(i, self.UNK) for i in sequence_list]
        if add_eos:
            _sequence_index += [self.EOS]
        if max_len is not None:
            sequence_index = [self.PAD] * max_len
            sequence_index[:seq_len] = _sequence_index
            return sequence_index
        else:
            return _sequence_index

    def inverse_transform(self, sequence_index):
        result = []
        for i in sequence_index:
            if i == self.EOS:
                break
            result.append(self.index2word.get(int(i), self.UNK_TAG))
        return result


# 实例化，供后续调用
num_sequence = NumSequence()

if __name__ == '__main__':
    num_sequence = NumSequence()
    print(num_sequence.dict)
    print(num_sequence.index2word)
    print(num_sequence.transform("1231230", add_eos=True))
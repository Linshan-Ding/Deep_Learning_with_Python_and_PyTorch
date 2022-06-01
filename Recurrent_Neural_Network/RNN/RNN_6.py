import jieba

raw_text = """我爱上海
              她喜欢北京
              他在杭州工作"""
stoplist = [' ', '\n']

# 利用jieba进行分词
words = list(jieba.cut(raw_text))
# 过滤停用词，如空格，回车符\n等
words = [i for i in words if i not in stoplist]

# 去重，然后对每个词加上索引或给一个整数
word_to_ix = {i: word for i, word in enumerate(set(words))}

from torch import nn
import torch

embeds = nn.Embedding(10, 8)
lists = []
for k, v in word_to_ix.items():
    tensor_value = torch.tensor(k)
    lists.append((embeds(tensor_value).data))

print(lists)
# coding = utf-8
'''
hello ,good day！
date: 2020/8/27
desc: 

'''

# Example1
# 文档 1：this is the bayes document；
# 文档 2：this is the second second document；
# 文档 3：and the third one；
# 文档 4：is this the document。
# 计算文档里都有哪些单词，这些单词在不同文档中的 TF-IDF 值是多少呢？

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vec = TfidfVectorizer()

documents = [
    'this is the bayes document',
    'this is the second second document',
    'and the third one',
    'is this the document'
]
tfidf_matrix = tfidf_vec.fit_transform(documents)

print(tfidf_matrix)
# feature_names按照字母顺序排列
print('不重复的词:', tfidf_vec.get_feature_names())
# 每个单词的ID: {'this': 8, 'is': 3, 'the': 6, 'bayes': 1, 'document': 2, 'second': 5, 'and': 0, 'third': 7, 'one': 4}
# ID 按照上面数组中的顺序排列
print('每个单词的ID:', tfidf_vec.vocabulary_)
# tfidf值按照feature_names数组中的单词顺序排列，一共4行9列
print('每个单词的tfidf值:', tfidf_matrix.toarray())

# a = tfidf_matrix.toarray()
# print(a.shape)
# 二项分布的试验结果只有两个(成功和失败)，而多项分布的试验结果则多于两个。
# Example2

# 中文文本分类
import os
import jieba
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

warnings.filterwarnings('ignore')

'''
text = open('text classification/train/女性/1.txt', 'r', encoding='gb18030').read()
print(text)
textcut = jieba.cut(text)
text_with_spaces = ''
for word in textcut:
    text_with_spaces += word + ' '
print(text_with_spaces)
'''

def cut_words(file_path):
    """
        对文本进行切词
        :param file_path: txt文本路径
        :return: 用空格分词的字符串
    """
    text = open(file_path, 'r', encoding='gb18030').read()
    textcut = jieba.cut(text)
    text_with_spaces = ''
    for word in textcut:
        text_with_spaces += word + ' '
    return text_with_spaces

def loadfile(file_dir, label):
    """
        将路径下的所有文件加载
        :param file_dir: 保存txt文件目录
        :param label: 文档标签
        :return: 分词后的文档列表和标签
        """
    file_list = os.listdir(file_dir)
    words_list = []
    labels_list = []
    for file in file_list:
         file_path = file_dir +'/'+ file
         words_list.append(cut_words(file_path))
         labels_list.append(label)
    return words_list, labels_list

# 训练数据
train_words_list1, train_labels1 = loadfile('text classification/train/女性', '女性')
train_words_list2, train_labels2 = loadfile('text classification/train/体育', '体育')
train_words_list3, train_labels3 = loadfile('text classification/train/文学', '文学')
train_words_list4, train_labels4 = loadfile('text classification/train/校园', '校园')

train_words_list = train_words_list1 + train_words_list2 + train_words_list3 + train_words_list4
train_labels = train_labels1 + train_labels2 + train_labels3 + train_labels4

# 测试数据
test_words_list1, test_labels1 = loadfile('text classification/test/女性', '女性')
test_words_list2, test_labels2 = loadfile('text classification/test/体育', '体育')
test_words_list3, test_labels3 = loadfile('text classification/test/文学', '文学')
test_words_list4, test_labels4 = loadfile('text classification/test/校园', '校园')

test_words_list = test_words_list1 + test_words_list2 + test_words_list3 + test_words_list4
test_labels = test_labels1 + test_labels2 + test_labels3 + test_labels4

print(train_words_list)
print(train_labels)

stop_words = open('text classification/stop/stopword.txt', 'r', encoding='utf-8').read()
stop_words = stop_words.encode('utf-8').decode('utf-8-sig') # 列表头部\ufeff处理
stop_words = stop_words.split('\n') # 根据分隔符分隔

print(stop_words)

# 计算单词权重
tf = TfidfVectorizer(stop_words=stop_words, max_df=0.5)
train_features = tf.fit_transform(train_words_list)
print(train_features)
b=train_features.toarray()
print(b.shape)
type(train_features)

# 上面fit过了，这里transform
test_features = tf.transform(test_words_list)

# 多项式贝叶斯分类器
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB(alpha=0.001).fit(train_features, train_labels)
predicted_labels=clf.predict(test_features)

# 计算准确率
print('准确率为：', metrics.accuracy_score(test_labels, predicted_labels))
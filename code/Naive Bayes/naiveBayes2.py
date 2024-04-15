'''
@File    :     naiveBayes2.py    
@Contact :     zhangzhilong2022@gmail.com
@Modify Time   2024/3/8 11:04   
@Author        huahai2022
@Desciption
'''
import jieba
import re

from sklearn import model_selection
from sklearn.naive_bayes import MultinomialNB


def text_to_words(file_path):
    '''
    分词
    return:sentences_arr, lab_arr
    '''
    sentences_arr = []
    lab_arr = []
    with open(file_path,'r',encoding='utf8') as f:
        for line in f.readlines():
            lab_arr.append(line.split('_!_')[1])	#得到标签
            sentence = line.split('_!_')[-1].strip()	#得到句子
            sentence = re.sub("[\s+\.\!\/_,$%^*(+\"\')]+|[+——()?【】“”！，。？、~@#￥%……&*（）《》：]+", "",sentence) #去除标点符号
            sentence = jieba.lcut(sentence, cut_all=False)	#切分句子
            sentences_arr.append(sentence)
    return sentences_arr, lab_arr

def load_stopwords(file_path):
    '''
    创建停用词表
    参数 file_path:停用词文本路径
    return：停用词list
    '''
    stopwords = [line.strip() for line in open(file_path, encoding='UTF-8').readlines()]
    return stopwords



def get_dict(sentences_arr,stopswords):
    '''
    遍历数据，去除停用词，统计词频
    return: 生成词典
    '''
    word_dic = {}
    for sentence in sentences_arr:
        for word in sentence:
            if word != ' ' and word.isalpha():
                if word not in stopswords:
                    word_dic[word] = word_dic.get(word,1) + 1
    word_dic=sorted(word_dic.items(),key=lambda x:x[1],reverse=True) #按词频序排列

    return word_dic


def get_feature_words(word_dic,word_num):
    '''
    从词典中选取N个特征词，形成特征词列表
    return: 特征词列表
    '''
    n = 0
    feature_words = []
    for word in word_dic:
        if n < word_num:
            feature_words.append(word[0])
        n += 1
    return feature_words


# 文本特征
def get_text_features(train_data_list, test_data_list, feature_words):
    '''
    根据特征词，将数据集中的句子转化为特征向量
    '''
    def text_features(text, feature_words):
        text_words = set(text)
        features = [1 if word in text_words else 0 for word in feature_words] # 形成特征向量
        return features
    train_feature_list = [text_features(text, feature_words) for text in train_data_list]
    test_feature_list = [text_features(text, feature_words) for text in test_data_list]
    return train_feature_list, test_feature_list





#获取分词后的数据及标签
sentences_arr, lab_arr = text_to_words('D:\project\\tutorial\机器学习\code\\Naive Bayes\\news_classify_data.txt')
print(sentences_arr[:5])
#加载停用词
stopwords = load_stopwords('D:\project\\tutorial\机器学习\code\\Naive Bayes\\stopwords_cn.txt')
# 生成词典
word_dic = get_dict(sentences_arr,stopwords)
#生成特征词列表
feature_words =  get_feature_words(word_dic,10000)
#数据集划分
train_data_list, test_data_list, train_class_list, test_class_list = model_selection.train_test_split(sentences_arr, lab_arr, test_size=0.2)
train_feature_list,test_feature_list = get_text_features(train_data_list,test_data_list,feature_words)
print(train_feature_list[:5])
print(len(train_feature_list))
from sklearn.metrics import accuracy_score,classification_report
#获取朴素贝叶斯分类器
classifier = MultinomialNB(alpha=1.0,  # 拉普拉斯平滑
                          fit_prior=True,  #否要考虑先验概率
                          class_prior=None)
#进行训练
classifier.fit(train_feature_list, train_class_list)
# 在验证集上进行验证
predict = classifier.predict(test_feature_list)
test_accuracy = accuracy_score(predict,test_class_list)
print("accuracy_score: %.4lf"%(test_accuracy))
print("Classification report for classifier:\n",classification_report(test_class_list, predict))
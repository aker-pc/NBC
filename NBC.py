import pandas as pd
import jieba
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.metrics import classification_report

# 导入文件
file = open(r"ChnSentiCorp_htl_all.csv", 'r', encoding='utf-8')
# 依次读取每行
reviews = file.readlines()
data = []
# 用嵌套list的方式来存储数据
for r in reviews:
    data.append([r[0], r[2:]])

d1 = pd.DataFrame(data)
# 更改默认的显示长度
pd.set_option('max_colwidth', 200)
#
d1.columns = ['two_class', 'comment']  # 修改列名


# head()函数默认输出前五行的数据，可通过添加head括号中的值，来更改默认值
# print('修改列名后的数据（只显示前5行）：\n' + str(d1.head()))
# print(d1.shape)


def word_clean(mytext):
    return ' '.join(jieba.lcut(mytext))


x = d1[['comment']]
# apply函数遍历每个元素，对其执行指定的函数
x['cutted_comment'] = x.comment.apply(word_clean)
print(x['cutted_comment'] )
# print(x.shape)
# # 查看分词后的结果
# print('数据分词后的结果：\n' + str(x.cutted_comment[:5]))
# y = d1.two_class
# # print(y.shape)
# # 数据按3:1分为训练集和测试集
# # test_size 缺省时，默认该值
# x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)
# # print('训练集：' + str(x_train.shape) + ' ' + str(y_train.shape))
# print('测试集：' + str(x_test.shape) + ' ' + str(y_test.shape))

#
# def get_custom_stopwords(fpath):
#     with open(fpath, encoding='utf-8') as f:
#         stopwords = f.read()
#     stopwords_list = stopwords.split('\n')
#     return stopwords_list
#
#
stop_words = get_custom_stopwords(r'./cn_stopwords.txt')
# # # 不去停用词
# # 转化为由TF-IDF表达的权重信息构成的向量，创建向量模型
# # vect = TfidfVectorizer(encoding='latin-1')
# # # 相当于先fit(完成语料分析,提取词典),再transform(把每篇文档转化为向量以构成矩阵)
# # term_matrix = pd.DataFrame(vect.fit_transform(x_train.cutted_comment).toarray(), columns=vect.get_feature_names())
# # print(term_matrix)
# # print('原始的特征数量：' + str(term_matrix.shape))
# # # 去除停用词
# # vect = TfidfVectorizer(encoding='latin-1', stop_words=frozenset(stop_words))
# # term_matrix = pd.DataFrame(vect.fit_transform(x_train.cutted_comment).toarray(), columns=vect.get_feature_names())
# # print(term_matrix)
# # print('去掉停用词的特征数量：' + str(term_matrix.shape))
# #
# max_df = 0.8  # 在超过这一比例的文档中出现的关键词（过于平凡），去除掉。
# min_df = 3  # 在低于这一数量的文档中出现的关键词（过于独特），去除掉。
# vect = TfidfVectorizer(max_df=max_df, min_df=min_df, stop_words=frozenset(stop_words))
# term_matrix = pd.DataFrame(vect.fit_transform(x_train.cutted_comment).toarray(), columns=vect.get_feature_names())
# # print('进一步处理后的特征数量：' + str(term_matrix.shape))
# # print(term_matrix)
# #
# # 使用贝叶斯预测分类
# nb = MultinomialNB(alpha=0.001)
# # 利用管道顺序连接工作
# pipe = make_pipeline(vect, nb)
# #
# # # 交叉验证的准确率
# # cross_result = cross_val_score(pipe, x_train.cutted_comment, y_train, cv=5, scoring='accuracy').mean()
# # print('交叉验证的准确率：' + str(cross_result))
# #
# # # 进行预测
# pipe.fit(x_train.cutted_comment, y_train)
# y_pred = pipe.predict(x_test.cutted_comment)
# # print(type(x_train.cutted_comment))
# # print(type(x_test.cutted_comment))
# # print(x_train.cutted_comment)
# # print(y_train)
# # print("预测结果：", y_pred, "共预测：", len(y_pred))
# # # 准确率测试
# accuracy = metrics.accuracy_score(y_test, y_pred)
# # print('准确率：' + str(accuracy))
# # # 混淆矩阵
# # print('混淆矩阵：' + str(metrics.confusion_matrix(y_test, y_pred)))
# # print("朴素贝叶斯分类详情", pipe, "分类表现如下")
# # print(classification_report(y_test, y_pred))
# lan = '这是 货真价实 的 好 酒店 ！ 虽然 硬件 很 简单 ， 但是 大部分 软件 都 非常 棒 ！ 服务 、 卫生 非常 好 ！ 环境 、 位置 好 ！ 下次 还 住 这里 ！ 我会 向 朋友 推荐 的 ！ 早餐 再 改进 一些 就 更好 啦 ！ \n'
# se = pd.Series(lan)
# new = pipe.predict(se)
# #
# print(new)

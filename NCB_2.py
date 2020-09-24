from os import path
from PIL import Image

from cv2 import imread
import pandas as pd
from wordcloud import WordCloud, ImageColorGenerator
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import jieba
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier


def load_data(file_path):
    file = open(file_path, 'r', encoding='utf-8')
    file_line = file.readlines()
    data = []
    for i in file_line:
        # 按照str位置进行的切片
        data.append([i[0], i[2:]])
    return data


def data_option(data, max_col, col_name: list):
    """
    :param data: 待修改格式的数据
    :param max_col: Dataframe最大显示数字
    :param col_name: 列名
    :return:
    """
    pd.set_option('max_colwidth', max_col)
    data.columns = col_name


def cut_word(data_pub_1):
    # def cut_word_in(mytext):
    #     return ' '.join(jieba.lcut(mytext))
    #
    # x = data_pub_1[['comment']]
    # x['cutted_comment'] = x.comment.apply(cut_word_in)

    # 生成
    # 1.读入txt文本数据
    text = str(data_pub_1[['comment']])
    # print(text)
    # 2.结巴中文分词，生成字符串，默认精确模式，如果不通过分词，无法直接生成正确的中文词云
    cut_text = jieba.cut(text)
    # print(type(cut_text))
    # 必须给个符号分隔开分词结果来形成字符串,否则不能绘制词云
    result = " ".join(cut_text)
    # print(result)

    # mask = imread("boy.png")
    # with open("alice.txt", "r") as file:
    #     txt = file.read()
    mask = imread("img.png")
    word = WordCloud(background_color="white", \
                     width=800, \
                     height=800,
                     font_path='msyh.ttc',
                     mask=mask,
                     ).generate(result)
    word.to_file('test.png')
    print("词云图片已保存")

    plt.imshow(word)  # 使用plt库显示图片
    plt.axis("off")
    plt.show()


def get_stopword(fpath):
    with open(fpath, encoding='utf-8') as f:
        stopwords = f.read()
    stopwords_list = stopwords.split('\n')
    return stopwords_list


def tf_dif_modle(label, dataset_cut, stop_sym, max_data, min_data, flag):
    """
    :param label:   评论性质
    :param dataset_cut: 评论数据
    :param stop_sym:    是否去除停用词标签 1 不去除， 否则去除
    :param max_data:    平凡度，超过这一平凡度，则去除这些词汇
    :param min_data:    独特度，超过这一独特度，则去除这些词汇
    :param flag:        1:Byes 2:随机森林 3：决策树 4：SVM 5：AdaBoost
    :return:
    """
    x_train, x_test, y_train, y_test = train_test_split(dataset_cut, label, random_state=1)
    stop_words = get_stopword(r'./cn_stopwords.txt')
    if stop_sym == 1:
        vect = TfidfVectorizer(encoding='lantin-1')
        term_matrix = pd.DataFrame(vect.fit_transform(x_train.cutted_comment).toarray(),
                                   columns=vect.get_feature_names())
    else:
        max_df = max_data  # 在超过这一比例的文档中出现的关键词（过于平凡），去除掉。
        min_df = min_data  # 在低于这一数量的文档中出现的关键词（过于独特），去除掉。
        vect = TfidfVectorizer(max_df=max_df, min_df=min_df, stop_words=frozenset(stop_words))
        term_matrix = pd.DataFrame(vect.fit_transform(x_train.cutted_comment).toarray(),
                                   columns=vect.get_feature_names())
    return term_matrix
    # if flag == 1:
    #     model_pipe = Naive_Byes(vect)
    #     model_pipe.fit(x_train.cutted_comment, y_train)
    # elif flag == 2:
    #     model_pipe = Random_forest(vect)
    #     model_pipe.fit(x_train.cutted_comment, y_train)
    # elif flag == 3:
    #     model_pipe = Decision_tree(vect)
    #     model_pipe.fit(x_train.cutted_comment, y_train)
    # elif flag == 4:
    #     model_pipe = SVM_fun(vect)
    #     model_pipe.fit(x_train.cutted_comment, y_train)
    # elif flag == 5:
    #     model_pipe = AdaBoost(vect)
    #     model_pipe.fit(x_train.cutted_comment, y_train)
    # return term_matrix, model_pipe, x_train, x_test, y_train, y_test


def Naive_Byes(word_vect):
    nb = MultinomialNB(alpha=0.001)
    pipe = make_pipeline(word_vect, nb)
    return pipe


def Random_forest(word_vect):
    Rlf = RandomForestClassifier(n_estimators=100, max_depth=20, min_samples_split=2, random_state=12)
    pipe = make_pipeline(word_vect, Rlf)
    return pipe


def Decision_tree(word_vect):
    Dct = DecisionTreeClassifier(criterion="entropy")
    pipe = make_pipeline(word_vect, Dct)
    return pipe


def SVM_fun(word_vect):
    SVM = SVC(gamma='scale')
    pipe = make_pipeline(word_vect, SVM)
    return pipe


def AdaBoost(word_vect):
    Ada = AdaBoostClassifier(n_estimators=100)
    pipe = make_pipeline(word_vect, Ada)
    return pipe


def evaluation_fun(pipe, train_x, train_y, test_x, test_y):
    cross_result = cross_val_score(pipe, train_x.cutted_comment, train_y, cv=5, scoring='accuracy').mean()
    print('交叉验证的准确率：' + str(cross_result))
    pre_y = pipe.predict(test_x.cutted_comment)
    accuracy = metrics.accuracy_score(test_y, pre_y)
    print('准确率：' + str(accuracy))
    print("分类详情", pipe, "分类表现如下")
    print(classification_report(test_y, pre_y))


if __name__ == '__main__':
    data_pub = load_data('ChnSentiCorp_htl_all.csv')
    data_pub = pd.DataFrame(data_pub)

    # 列名(暂不可更改)
    pd_col_name = ['two_class', 'comment']
    data_option(data_pub, 200, pd_col_name)

    # 分词后的评论数据集
    cut_word(data_pub)
    # data_cut = cut_word(data_pub)
    # data_label = data_pub.two_class

    # 训练模型
    # 模型选择参数（可选范围1~5）
    # model = 5
    # freq_matrix = tf_dif_modle(data_label, data_cut, 1, 0.8, 3, model)
    # print(freq_matrix)
    # freq_matrix = np.array(freq_matrix)
    # count = 0
    # for i in freq_matrix[:, 2]:
    #     if i == 1:
    #         count += 1
    # print(count)
    # a = freq_matrix.iloc[:, :]
    # freq_matrix.to_excel('./a.xlsx')
    # print(type(freq_matrix))
    # print(freq_matrix.iloc[:, ])
    # freq_matrix, modle_fit, x_train_act, x_test_act, y_train_act, y_test_act = tf_dif_modle(data_label, data_cut, 1, 0.8, 3, model)
    # pre = modle_fit.predict(x_train_act.cutted_comment)
    # print(pre)
    # 评价模型
    # evaluation_fun(modle_fit, x_train_act, y_train_act, x_test_act, y_test_act)

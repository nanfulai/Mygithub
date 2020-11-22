#  The code in this file
#  is for Part 1 of the tutorial on Natural Language Processing.
#
# *************************************** #

import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from KaggleWord2VecUtility import KaggleWord2VecUtility
import pandas as pd
import numpy as np

import nltk
nltk.download('stopwords')
import nltk
nltk.download('punkt')

if __name__ == '__main__':
    train = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'labeledTrainData.tsv'), header=0, \
                    delimiter="\t", quoting=3)
    test = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'testData.tsv'), header=0, delimiter="\t", \
                   quoting=3 )

    print('The first review is:')
    print(train["review"][0])

    input("Press Enter to continue...")


    print('Download text data sets. If you already have NLTK datasets downloaded, just close the Python download window...')
    #nltk.download()  # Download text data sets, including stop words

    # 初始化一个空列表
    clean_train_reviews = []
    print("Cleaning and parsing the training set movie reviews...\n")
    for i in range( 0, len(train["review"])):
        clean_train_reviews.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(train["review"][i], True)))


    # ****** 从训练集中创建一个词袋
    #
    print("Creating the bag of words...\n")

    vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 5000)

    # fit_transform（）有两个功能：首先，它适合模型
    #并学习词汇；其次，它改变了我们的训练数据
    #进入特征向量。fit_transform的输入应为以下内容的列表
    #个字符串。
    train_data_features = vectorizer.fit_transform(clean_train_reviews)
    np.asarray(train_data_features)

    # *******用词袋模型训练随机森林
    print("Training the random forest (this may take a while)...")


    # 初始化具有100棵树的随机森林分类器
    forest = RandomForestClassifier(n_estimators = 100)

    # 使森林适应训练集，使用词袋作为
    # 功能和情感标签作为响应变量

    forest = forest.fit( train_data_features, train["sentiment"] )



    # 创建一个空列表，并将干净的评论一个接一个地添加
    clean_test_reviews = []

    print("Cleaning and parsing the test set movie reviews...\n")
    for i in range(0,len(test["review"])):
        clean_test_reviews.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(test["review"][i], True)))

    test_data_features = vectorizer.transform(clean_test_reviews)
    np.asarray(test_data_features)

   # 使用随机森林对感情标签进行预测
    print("Predicting test labels...\n")
    result = forest.predict(test_data_features)

    output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )

    # 使用pandas编写以逗号分隔的输出文件
    output.to_csv(os.path.join(os.path.dirname(__file__), 'data', 'Bag_of_Words_model.csv'), index=False, quoting=3)
    print("Wrote results to Bag_of_Words_model.csv")
# The code in this file
#  is for Parts 2 and 3 of the tutorial, which cover how to
#  train a model using Word2Vec.
#
# *************************************** #


# ******导入训练集和测试
#
import pandas as pd
import os
from nltk.corpus import stopwords
import nltk.data
import logging
import numpy as np
from gensim.models import Word2Vec
from sklearn.ensemble import RandomForestClassifier

from KaggleWord2VecUtility import KaggleWord2VecUtility


# ****** Define functions to create average word vectors
#

def makeFeatureVec(words, model, num_features):
    # Function to average all of the word vectors in a given
    # paragraph
    #
    # Pre-initialize an empty numpy array (for speed)
    featureVec = np.zeros((num_features,), dtype="float32")
    #
    nwords = 0.
    #
    # Index2word是一个列表，其中包含以下单词的名称
    #  模型的词汇量。将其转换为一组以提高速度
    index2word_set = set(model.wv.index2word)
    #
    # 遍历评论中的每个单词，如果该单词在模型的
    # vocaublary，将其特征向量添加到总数中
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1.
            featureVec = np.add(featureVec, model[word])
    #
    # 将结果除以字数即可得出平均值
    featureVec = np.divide(featureVec, nwords)
    return featureVec


def getAvgFeatureVecs(reviews, model, num_features):

    counter = 0.
    #给定一组评论（每个评论一个单词列表），计算
    #每个特征向量的平均特征向量，并返回2D
    #numpy数组
    reviewFeatureVecs = np.zeros((len(reviews), num_features), dtype="float32")
    #
    # Loop through the reviews
    for review in reviews:
        #
        # Print a status message every 1000th review
        if counter % 1000. == 0.:
            print(
            "Review %d of %d" % (counter, len(reviews)))
        #
        # Call the function (defined above) that makes average feature vectors
        reviewFeatureVecs[int(counter)] = makeFeatureVec(review, model, \
                                                         num_features)
        #
        # Increment the counter
        counter = counter + 1.
    return reviewFeatureVecs


def getCleanReviews(reviews):
    clean_reviews = []
    for review in reviews["review"]:
        clean_reviews.append(KaggleWord2VecUtility.review_to_wordlist(review, remove_stopwords=True))
    return clean_reviews


if __name__ == '__main__':

    # Read data from files
    train = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'labeledTrainData.tsv'), header=0,
                        delimiter="\t", quoting=3)
    test = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'testData.tsv'), header=0, delimiter="\t",
                       quoting=3)
    unlabeled_train = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', "unlabeledTrainData.tsv"), header=0,
                                  delimiter="\t", quoting=3)

    # Verify the number of reviews that were read (100,000 in total)
    print(
    "Read %d labeled train reviews, %d labeled test reviews, " \
    "and %d unlabeled reviews\n" % (train["review"].size,
                                    test["review"].size, unlabeled_train["review"].size))

    # Load the punkt tokenizer
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    # ****** Split the labeled and unlabeled training sets into clean sentences
    #
    sentences = []  # Initialize an empty list of sentences

    print(
    "Parsing sentences from training set")
    for review in train["review"]:
        sentences += KaggleWord2VecUtility.review_to_sentences(review, tokenizer)

    print(
    "Parsing sentences from unlabeled set")
    for review in unlabeled_train["review"]:
        sentences += KaggleWord2VecUtility.review_to_sentences(review, tokenizer)


    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', \
                        level=logging.INFO)

    # 设置各种参数的值
    num_features = 300  # 词向量维数
    min_word_count = 40
    num_workers = 4  #并行运行的线程数
    context = 10
    downsampling = 1e-3  #下采样设置

    print("Training Word2Vec model...")
    model = Word2Vec(sentences, workers=num_workers, \
                     size=num_features, min_count=min_word_count, \
                     window=context, sample=downsampling, seed=1)


    model.init_sims(replace=True)


    model_name = "300features_40minwords_10context"
    model.save(model_name)

    model.doesnt_match("man woman child kitchen".split())
    model.doesnt_match("france england germany berlin".split())
    model.doesnt_match("paris berlin london austria".split())
    model.most_similar("man")
    model.most_similar("queen")
    model.most_similar("awful")

    # ****** Create average vectors for the training and test sets
    #
    print(
    "Creating average feature vecs for training reviews")

    trainDataVecs = getAvgFeatureVecs(getCleanReviews(train), model, num_features)

    print(
    "Creating average feature vecs for test reviews")

    testDataVecs = getAvgFeatureVecs(getCleanReviews(test), model, num_features)


    # 使用100棵树为训练数据拟合一个随机森林
    forest = RandomForestClassifier(n_estimators=100)

    print("Fitting a random forest to labeled training data...")
    forest = forest.fit(trainDataVecs, train["sentiment"])

    # Test & extract results
    result = forest.predict(testDataVecs)

    # Write the test results
    output = pd.DataFrame(data={"id": test["id"], "sentiment": result})
    output.to_csv("Word2Vec_AverageVectors.csv", index=False, quoting=3)
    print("Wrote Word2Vec_AverageVectors.csv")

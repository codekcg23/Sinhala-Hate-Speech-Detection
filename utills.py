import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import sklearn.metrics as metrics
#from sklearn.metrics import accuracy_score, f1_score, precision_score,roc_curve,roc_auc_score,confusion_matrix,recall_score
#from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import re
import gensim
import emoji


def sentenceSeperator(txt):
    return re.sub(r'[\.?;!,\u2022]', r'\g<0> ', txt)


def removePunctuationSpecial(txt):
    exclude = set(
        ".…,‌ ¸‚\"/|—¦”‘\'“’´`<>#@$%^=&*+\-£˜()\[\]{\}:;–_‪‬‏")

    return ''.join([(c if c not in exclude else " ") for c in txt])


def removePunctuation(txt):
    """
    replace URL of a text
    :param text: text to replace urls
    :return: url removed text
    """
    # set(",.:;'\"-/´`%")
    exclude = set(
        ".…,‌ ¸‚\"/|—¦”‘\'“’´!?<>@#$%^=&*+\-£˜()\[\]{\}:;–_")

    return ''.join([(c if c not in exclude else " ") for c in txt])


def removeSpecialCharacters(txt):
    exclude = set(
        "Ê�\u00a0\u2003\ufffd\uf020\uf073\uf06c\uf190\u202a\u202c\u200f\u200c\u0160\u00ad\u0088\uf086\u200b\ufeff")

    return ''.join([(c if c not in exclude else '') for c in txt])
# removing digits


def removeNumber(txt):
    """
    replace number of a text
    :param text: text to replace number
    :return: numbers removed text
    """
    return ''.join(c for c in txt if not c.isnumeric())

# Remove emojis


def removeEmoji(txt):
    """
    replace emoji of a text with ''
    :param text: text to replace emojis
    :return: emoji removed text
    """
    import emoji
    return emoji.get_emoji_regexp().sub(u'', txt)

# def replaceEmoji(txt):
#     print()


def removeUrl(txt):
    """
    replace URL of a text
    :param text: text to replace urls
    :return: url removed text
    """
    import re
    return re.sub(r'(http://www\.|https://www\.|http://|https://)[a-z0-9]+([\-.]{1}[a-z0-9A-Z/]+)*', '', txt)


def removeRetweetState(txt):
    """
    remove retweet states in the beginning such as "RT @sam92ky: "
    :param text: text
    :return: text removed retweets state
    """

    return re.sub(r'RT @\w*: ', '', txt)


def removeMention(txt):
    """
    replace @username of a text
    :param text: text to replace username
    :return: username removed text
    """
    return re.sub(r'@[a-zA-Z]+', '', txt)
    # return re.sub(r'@\w*', '', txt)


def removeEnglishWords(txt):

    return re.sub(r'[a-zA-Z]+', '', txt)


def removeSentenceContainsEnglish(df, col):
    """
    remove rows contains english letter string of a dataframe
    :param df: Name of dataframe
    :param col : Column inclding text
    :return: Dataframe contains non englsih charachters
    """

    print("Input dataframe size = ", len(df))
    for s in df[col]:
        english_list = re.findall(r'[a-zA-Z]+', s)
        # print(english_list)
        if (english_list != []):
            i = df[df[col] == s].index
            df.drop(i, axis=0, inplace=True)
            # print(s)
    print("Cleaned dataframe size - removed Strings contain Englishs letters ", len(df))
    return df

# remove stop words


# def removeStopWords(txt, stop_words):
#     #lst_text = text.split()
#     #lst_text = [word for word in lst_text if word not in lst_stopwords]
#     return ''.join([(w if w not in stop_words else " ") for w in txt])


def identifySinhalaText(txt):

    sinhala_list = re.findall(r'[\u0D80-\u0DFF]+', txt)
    return sinhala_list
# def ignore_characters(txt):
#     return ' '.join(c for c in txt if not c.startswith('\u'))


def stemWords(txt):
    from sinling import SinhalaStemmer
    stemmer = SinhalaStemmer()

    #stemmed = stemmer.stem(word)[0]
    return ' '.join([stemmer.stem(w)[0] for w in txt.split()])
    #   text =[]
    # for word in txt.split():
    #     #print(word)
    #     if len(word) < 4:
    #         word = word

    #     # remove 'ට'
    #     if word[-1] == 'ට':
    #         word =word[:-1]

    #     # remove 'ද'
    #     elif word[-1] == 'ද':
    #         word =word[:-1]

    #     # remove 'ටත්'
    #     elif word[-3:] == 'ටත්':
    #         word = word[:-3]

    #     # remove 'එක්'
    #     elif word[-3:] == 'ෙක්':
    #         word = word[:-3]

    #     # remove 'එ'
    #     elif word[-1:] == 'ෙ':
    #         word = word[:-1]

    #     # remove 'ක්'
    #     elif word[-2:] == 'ක්':
    #         word =word[:-2]

    #     # remove 'ගෙ' (instead of ගේ because this step comes after simplifying text)
    #     elif word[-4:] == 'ගේ':
    #         word = word[:-4]
    #     elif word[-3:] == 'ෙන්':
    #         word =word[:-3]
    #     text.append(word)
    # print(text)
    # return ' '.join(text)


def custom_stemmer_shorterSuffix(txt):
    # custom stemmer longest prefix len >5
    import sys
    sys.path.insert(1, 'G:\\Github\\Sinhala-Hate-Speech-Detection')
    import sinhala_stemmer
    stemmer = sinhala_stemmer.SinhalaStemmer()
    return ' '.join([stemmer.stem(w)[0] for w in txt.split()])


def custom_stemmer_longerSuffix(txt):
    import sys
    sys.path.insert(1, 'G:\\Github\\Sinhala-Hate-Speech-Detection')
    import sinhala_stemmer
    # custom stemmer longest prefix len >5
    stemmer = sinhala_stemmer.SinhalaStemmer()
    return ' '.join([stemmer.stem(w, True)[0] for w in txt.split()])

# def text_normalize():


def removeStopWords(txt):
    """
    remove stop words
    """
    stop_words = List = open(
        'G:\Github\Sinhala-Hate-Speech-Detection\Datasets\stop_words.txt', encoding='utf-8').read().splitlines()
    remove_stop_words = ['ඕහෝ', 'අනේ', 'අඳෝ', 'අපොයි', 'අපෝ', 'අයියෝ',
                         'ආයි', 'ඌයි', 'චී', 'චිහ්', 'චික්', 'නෑ', 'එම්බා', 'එම්බල', 'බොල']
    for w in remove_stop_words:
        if w in stop_words:
            stop_words.remove(w)
    return ' '.join([(w if w not in stop_words else ' ') for w in txt.split()])


def removeUnicode(txt):
    exclude = set(
        "\u200c\u200b\u2063")

    return ''.join([(c if c not in exclude else '') for c in txt])
# def POS_tags():

# def NER():


def preprocessor(df, col, seperator=True, url=True, mention=True, number=True, non_sinhala=True, special_characters=True, puntuation=True, puntuation_special=False, emoji_remove=True,  stop_word=False, stem=False, custom_stem_shorter=False, custom_stem_longer=True):

   # if(tokenize):
    df['cleaned'] = df[col]
    if(seperator):
        df['cleaned'] = df['cleaned'].apply(lambda x: sentenceSeperator(x))
        print('seperator')
    if(url):
        df['cleaned'] = df['cleaned'].apply(lambda x: removeUrl(x))
        print('-url-')
    if(mention):
        df['cleaned'] = df['cleaned'].apply(lambda x: removeMention(x))
        print('-mention-')
    if(non_sinhala):
        df['cleaned'] = df['cleaned'].apply(lambda x: removeEnglishWords(x))
        print('-non_sinhala-')
    if(special_characters):
        df['cleaned'] = df['cleaned'].apply(
            lambda x: removeSpecialCharacters(x))
        print('-special charactrer-')
    if(puntuation):
        df['cleaned'] = df['cleaned'].apply(lambda x: removePunctuation(x))
        print('-puntuation-')
    if(puntuation_special):
        df['cleaned'] = df['cleaned'].apply(
            lambda x: removePunctuationSpecial(x))
        print('-puntuation_special-')
    if(emoji_remove):
        df['cleaned'] = df['cleaned'].apply(lambda x: removeEmoji(x))
        print('-emoji remove-')
    # if(emoji_replace):
    if(number):
        df['cleaned'] = df['cleaned'].apply(lambda x: removeNumber(x))
        print('-number-')
    if(stop_word):
        df['cleaned'] = df['cleaned'].apply(lambda x: removeStopWords(x))
        print('-stop word-')
    #df['cleaned'] = df['cleaned'].apply(lambda x: removeUnicode(x))
    if(stem):
        df['cleaned'] = df['cleaned'].apply(lambda x: stemWords(x))
        print('-stem-')
    if(custom_stem_shorter):
        df['cleaned'] = df['cleaned'].apply(
            lambda x: custom_stemmer_shorterSuffix(x))
        print('-custome stemmer shorter suffix-')
    if(custom_stem_longer):
        df['cleaned'] = df['cleaned'].apply(
            lambda x: custom_stemmer_longerSuffix(x))
        print('-custome stemmer longer suffix-')
    # if(text_normalize):

    return df

# def preprocessor(df,col, url=True,mention=False,non_sinhala=False,puntuation=False, puntuation_special = False, emoji_remove = False, emoji_replace= True, stop_word = False,stem=False,text_nomalize=False):


#     df['cleaned'] = df[col].apply(lambda x: removePunctuation(x))
#     # print(df['cleaned'].head(n=15))
#     df['cleaned'] = df['cleaned'].apply(lambda x: removeNumber(x))
#     # print(df['cleaned'].head(n=15))
#     df['cleaned'] = df['cleaned'].apply(lambda x: removeEmoji(x))
#     df['cleaned'] = df['cleaned'].apply(lambda x: removeUrl(x))
#     df['cleaned'] = df['cleaned'].apply(lambda x: removeRetweetState(x))
#     df['cleaned'] = df['cleaned'].apply(lambda x: removeMention(x))
#     # print(df['cleaned'].head(n=15))
#     return df


def preprocess(df, col):

    # print(df['cleaned'].head(n=15))
    df['cleaned'] = df[col].apply(lambda x: sentenceSeperator(x))
    df['cleaned'] = df['cleaned'].apply(lambda x: removeEmoji(x))
    df['cleaned'] = df['cleaned'].apply(lambda x: removeUrl(x))
    df['cleaned'] = df['cleaned'].apply(lambda x: removeUnicode(x))
    df['cleaned'] = df['cleaned'].apply(lambda x: removeRetweetState(x))
    df['cleaned'] = df['cleaned'].apply(lambda x: removeMention(x))
    df['cleaned'] = df['cleaned'].apply(lambda x: removeEnglishWords(x))
    df['cleaned'] = df['cleaned'].apply(lambda x: removeSpecialCharacters(x))
    df['cleaned'] = df['cleaned'].apply(lambda x: removePunctuation(x))
    # print(df['cleaned'].head(n=15))
    df['cleaned'] = df['cleaned'].apply(lambda x: removeNumber(x))
    # print(df['cleaned'].head(n=15))

    return df


def confusion_Matrix(y_test, y_pred):

    confusionMatrix = pd.crosstab(y_test, y_pred, rownames=[
                                  "Actual"], colnames=["Predicted"])
    print(confusionMatrix)
    confusionMatrix.plot.bar(stacked=True)
    plt.legend(title='mark')
    plt.show()


def PlotRocAuc(y_test, y_pred, color, model_name):
    import plotly.graph_objects as go
    from sklearn.metrics import roc_curve, roc_auc_score
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            name="TPR = FPR",
            line=dict(color="black", dash="dash")
        )
    )

    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_pred)
    fig.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            name=f"{model_name}(AUC={auc_score})",
            marker=dict(color=color)
        )
    )

    fig.update_layout(title="ROC curve",
                      xaxis_title="False Positive Rate",
                      yaxis_title="True Positive Rate")

    fig.show()


def prepare_dataset(df, name):
    df = preprocess(df, 'comment')
    print(name, len(df))
    X_train, X_test, Y_train, Y_test = train_test_split(
        df['cleaned'], df['label'], test_size=0.3, random_state=0, stratify=df['label'].values)
    print("X train {} Y train {} X test {} Y test {}".format(
        X_train.shape, Y_train.shape, X_test.shape, Y_test.shape))
    return (X_train, X_test, Y_train, Y_test)


def log_result(Y_test, Y_pred, name, df_name, feature_name, model_name,tag):

    import neptune
    from neptunecontrib.monitoring.metrics import expand_prediction, log_class_metrics, log_binary_classification_metrics, log_classification_report, log_confusion_matrix, log_prediction_distribution
    from neptunecontrib.api import log_table
    import os
    from dotenv import load_dotenv

    load_dotenv()
    NEPTUNE_PROJECT = os.getenv('NEPTUNE_PROJECT')
    NEPTUNE_API_TOKEN = os.getenv(('NEPTUNE_API_TOKEN'))
    neptune.init(project_qualified_name=NEPTUNE_PROJECT,
                 api_token=NEPTUNE_API_TOKEN)

    print("========= Eperiment - ", name, " =========")
    neptune.create_experiment(name)
    neptune.append_tag([tag,df_name, feature_name, model_name, name])

    log_class_metrics(Y_test, Y_pred)
    log_confusion_matrix(Y_test, Y_pred)
    log_classification_report(Y_test, Y_pred)


def classifier_feature(datasets, models, features):
    final_result = pd.DataFrame(
        columns=['Accuracy', 'F1-score', 'Recall', 'Precision', 'AUC'])
    for df_name, df in datasets.items():
        df_result = pd.DataFrame(
            columns=['Accuracy', 'F1-score', 'Recall', 'Precision', 'AUC'])
        #X_train, X_test, Y_train, Y_test = utills.prepare_dataset(df, df_name)
        X_train, X_test, Y_train, Y_test = train_test_split(
        df['cleaned'], df['label'], test_size=0.3, random_state=0, stratify=df['label'].values)
        for feature_name, feature in features.items():
            feature_result = pd.DataFrame(
                columns=['Accuracy', 'F1-score', 'Recall', 'Precision', 'AUC'])
            X_train_f, X_test_f = feature(X_train, X_test)
            for model_name, model in models.items():
                name = df_name + "+" + feature_name + "+" + model_name
                print(name)
                Y_pred = model(X_train_f, X_test_f, Y_train)
                accuracy, f1_score, recall, precision, auc = result(
                    Y_test, Y_pred)
                final_result.loc[name] = [
                    accuracy, f1_score, recall, precision, auc]
                feature_result.loc[model_name] = [
                    accuracy, f1_score, recall, precision, auc]
                key = model_name + "+" + feature_name
                df_result.loc[key] = [
                    accuracy, f1_score, recall, precision, auc]
                log_result(Y_test, Y_pred, name, df_name,
                           feature_name, model_name)
            print(" ==== ", feature_name, " ==== ")
            display(feature_result)
            log_table(feature_name, feature_result)
        print(" ==== ", df_name, " ==== ")
        display(df_result)
        log_table(df_name, df_result)
    display(final_result)
    log_table(name, final_result)


def result(y_test, y_pred):
    import sklearn.metrics as metrics
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    classes = np.unique(y_test)
    #y_test_array = pd.get_dummies(y_test, drop_first=False).values
    ## Accuracy, Precision, Recall
    accuracy = metrics.accuracy_score(y_test, y_pred)
    auc = metrics.roc_auc_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred)
    f1_score = metrics.f1_score(y_test, y_pred)

    print(metrics.classification_report(y_test, y_pred))

    # Plot confusion matrix
    cm = metrics.confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap=plt.cm.Blues, cbar=False)
    ax.set(xlabel="Predicted", ylabel="Actual", xticklabels=classes,
           yticklabels=classes, title="Confusion matrix")
    plt.yticks(rotation=0)
    plt.show()
#     utills.PlotRocAuc(y_test, y_pred, 'green', 'LR')

    return (accuracy, f1_score, recall, precision, auc)


def bow_word(X_train, X_test):
    bow = CountVectorizer(analyzer="word", tokenizer=lambda text: text.split(
    ), ngram_range=(1, 1), lowercase=False)
    bow.fit(X_train)
    X_train_bow = bow.transform(X_train)
    X_test_bow = bow.transform(X_test)

    # print(bow.get_feature_names()[:20])
    #print('The shape is', bow.shape)
    # postion
    # print(bow.vocabulary_)

    return X_train_bow, X_test_bow

# bow - char


def bow_char(X_train, X_test):
    bow = CountVectorizer(analyzer="char", ngram_range=(3, 5), lowercase=False)
    bow.fit(X_train)
    X_train_bow = bow.transform(X_train)
    X_test_bow = bow.transform(X_test)
    # X_train_bow = bow.fit_transform(X_train)
    # #X_train_bow = bow.transform(X_train)
    # X_test_bow = bow.transform(X_test)
    # print(bow.get_feature_names()[:20])
    #print('The shape is', tfidf.shape)
    # postion
    # print(bow.vocabulary_)

    return X_train_bow, X_test_bow

# TFIDF - word


def tfidf_word(X_train, X_test):
    tfidf = TfidfVectorizer(analyzer="word",  tokenizer=lambda text: text.split(
    ), ngram_range=(1, 1), lowercase=False)
    X_train_tfidf = tfidf.fit_transform(X_train)
    # print(X_train_tfidf.shape)
    #X_train_tfidf = tfidf.transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    # print(tfidf.get_feature_names()[:20])
    # #print('The shape is', tfidf.shape)
    # # postion
    # print(tfidf.vocabulary_)
    return X_train_tfidf, X_test_tfidf

# TFIDF - char


def tfidf_char(X_train, X_test):
    tfidf = TfidfVectorizer(
        analyzer="char", ngram_range=(3, 5), lowercase=False)
    X_train_tfidf = tfidf.fit_transform(X_train)
    #X_train_tfidf = tfidf.transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    # print(tfidf.get_feature_names()[:20])
    # #print('The shape is', tfidf.shape)
    # # postion
    # print(tfidf.vocabulary_)

    return X_train_tfidf, X_test_tfidf


def lr(X_train, X_test, Y_train):
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression()
    lr.fit(X_train, Y_train)
    Y_pred = lr.predict(X_test)
    return Y_pred


# SVC linear
def svc_linear(X_train, X_test, Y_train):
    from sklearn.svm import SVC
    svc = SVC(kernel="linear")
    svc.fit(X_train, Y_train)
    Y_pred = svc.predict(X_test)
    return Y_pred

# SVC rbf


def svc_rbf(X_train, X_test, Y_train):
    from sklearn.svm import SVC
    svc = SVC(kernel="rbf")
    svc.fit(X_train, Y_train)
    Y_pred = svc.predict(X_test)
    return Y_pred

# MNB


def MNB(X_train, X_test, Y_train):
    nb = MultinomialNB(alpha=0.01)
    nb.fit(X_train, Y_train)
    Y_pred = nb.predict(X_test)
    Y_prob = nb.predict_proba(X_test)[:, 1]
    return Y_pred

# Gaussian NB


def NB(X_train, X_test, Y_train):
    from sklearn.naive_bayes import GaussianNB
    nb = GaussianNB()
    nb = nb.fit(X_train, Y_train)
    Y_pred = nb.predict(X_test)
    Y_prob = nb.predict_proba(X_test)[:, 1]
    return Y_pred

# Random Forest


def RF(X_train, X_test, Y_train):
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(X_train, Y_train)
    Y_pred = rf.predict(X_test)
    Y_prob = rf.predict_proba(X_test)[:, 1]
    return Y_pred


def word_avg_vector(model,words_list):
    if len(words_list) < 1:  # whole sentence has no words or nan
        return np.zeros(300)
    else:
        vectorized = [model[word] if word in model else np.random.rand(300) for word in words_list]
    # doc = [word for word in doc if word in model.wv.index_to_key else np.random.rand(k)]
    return np.mean(vectorized, axis=0)

# TODO function for word_tfidf_avg_vector


def get_embedding(df,model):
    # TODO add option to select TFIDF vs mean embedding
    # avg vector
    embeddings = df.apply(lambda x: word_avg_vector(model,x))

    # tfidf weighted vector
    return list(embeddings)

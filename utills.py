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


def removePunctuation(txt):
    """
    replace URL of a text
    :param text: text to replace urls
    :return: url removed text
    """
    # set(",.:;'\"-/´`%")
    exclude = set(
        ".…,‌ ¸‚\"/|—¦”‘\'“’´!?<>#$%^=&*+\-£˜()\[\]{\}:;–Ê_�‪‬‏\u200b")

    return ''.join([(c if c not in exclude else " ") for c in txt])

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

    return emoji.get_emoji_regexp().sub(u'', txt)


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

    return re.sub(r'@\w*', 'PERSON', txt)


def removeEnglishWords(txt):

    return re.sub(r'[a-zA-Z\s]+', '', txt)


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


def removeStopWords(txt, stop_words):
    #lst_text = text.split()
    #lst_text = [word for word in lst_text if word not in lst_stopwords]
    return ''.join([(w if w not in stop_words else " ") for w in txt])


def identifySinhalaText(txt):

    sinhala_list = re.findall(r'[\u0D80-\u0DFF]+', txt)
    return sinhala_list
# def ignore_characters(txt):
#     return ' '.join(c for c in txt if not c.startswith('\u'))
# def stem():

# def stop_words():


# def POS_tags():

# def NER():


def preprocess(df, col):
    df['cleaned'] = df[col].apply(lambda x: removePunctuation(x))
    # print(df['cleaned'].head(n=15))
    df['cleaned'] = df['cleaned'].apply(lambda x: removeNumber(x))
    # print(df['cleaned'].head(n=15))
    df['cleaned'] = df['cleaned'].apply(lambda x: removeEmoji(x))
    df['cleaned'] = df['cleaned'].apply(lambda x: removeUrl(x))
    df['cleaned'] = df['cleaned'].apply(lambda x: removeRetweetState(x))
    df['cleaned'] = df['cleaned'].apply(lambda x: removeMention(x))
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


def removePunctuation(txt):
    """
    replace URL of a text
    :param text: text to replace urls
    :return: url removed text
    """
    # set(",.:;'\"-/´`%")
    exclude = set(
        ".…,‌ ¸‚\"/|—¦”‘\'“’´!?<>#$%^=&*+\-£˜()\[\]{\}:;–Ê_�‪‬‏")

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
    import emoji
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
    import re
    return re.sub(r'RT @\w*: ', '', txt)


def removeMention(txt):
    """
    replace @username of a text
    :param text: text to replace username
    :return: username removed text
    """
    import re
    return re.sub(r'@\w*', 'PERSON', txt)


def removeEnglishWords(txt):

    import re
    return re.sub(r'[a-zA-Z\s]+', '', txt)


def removeSentenceContainsEnglish(df, col):
    """
    remove rows contains english letter string of a dataframe
    :param df: Name of dataframe
    :param col : Column inclding text
    :return: Dataframe contains non englsih charachters
    """

    import re
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


def identifySinhalaText(txt):
    import re
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
    import matplotlib.pyplot as plt
    import pandas as pd
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


def removePunctuation(txt):
    # set(",.:;'\"-/´`%")
    exclude = set(
        ".…,‌ ¸‚\"/|—¦”‘\'“’´!?<>@#$%^=&*+\-£˜()\[\]{\}:;–Ê_�‪‬‏0123456789")

    return ''.join([(c if c not in exclude else " ") for c in txt])

# removing digits


def removeNumber(txt):
    return ''.join(c for c in txt if not c.isnumeric())

# Remove emojis


def removeEmoji(txt):
    import emoji
    return emoji.get_emoji_regexp().sub(u'', txt)


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

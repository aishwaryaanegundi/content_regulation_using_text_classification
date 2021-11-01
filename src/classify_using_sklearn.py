import numpy as np
import pandas as pd
import codecs
import jieba
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, f1_score, recall_score
from sklearn.ensemble import RandomForestClassifier

train_data = pd.read_csv('./data/KDC_train.tsv', sep='\t', header=0)
test_data = pd.read_csv('./data/KDC_test.tsv', sep='\t', header=0)
max_features = 3400

def segment(row):
    text = row['text']
    return ' '.join(jieba.cut(text, cut_all=False))

def balance_samples(data, feature, labels):
    sample_sizes = []
    for l in labels:
        size = (data[data[feature]==l].shape)[0]
        sample_sizes.append(size)
    min_sample_size = min(sample_sizes)
    balanced_data = pd.DataFrame()
    for l in labels:
        balanced_data = balanced_data.append(data[data[feature]==l].head(min_sample_size), ignore_index = True)
    return balanced_data

train_data['segmented_text'] = train_data.apply(segment, axis=1)
test_data['segmented_text'] = test_data.apply(segment, axis=1)

train_df, val_df = train_test_split(train_data, test_size = 0.1, random_state = 28) 
train_df = balance_samples(train_df,'status', train_df['status'].unique())

stop_words_df = pd.read_csv('./data/stop_words.csv')
chinese_stopwords = stop_words_df['words'].tolist()

cv = CountVectorizer(max_features=max_features, ngram_range=(1, 3), stop_words=chinese_stopwords)
train_X = cv.fit_transform(train_df['text'])
valid_X = cv.fit_transform(val_df['text'])

tfidf_transformer = TfidfTransformer()
train_X = tfidf_transformer.fit_transform(train_X)
valid_X = tfidf_transformer.fit_transform(valid_X)

train_y = train_df['status'].values
valid_y = val_df['status'].values

clf_svc = SVC(kernel='linear')
clf_svc = clf_svc.fit(train_X, train_y)

predicted_valid = clf_svc.predict(valid_X)
prediction_acc = np.mean(predicted_valid == valid_y)
prediction_f1_score = f1_score(valid_y, predicted_valid, average='weighted')
prediction_recall = recall_score(valid_y, predicted_valid, average='weighted')
print('Model Support Vector classfier: ','Accuracy: ', prediction_acc, 
      'F1: ', prediction_f1_score, 'Recall: ', prediction_recall)

clf_nb = MultinomialNB(fit_prior='true')
clf_nb = clf_nb.fit(train_X, train_y)

predicted_valid = clf_nb.predict(valid_X)
prediction_acc = np.mean(predicted_valid == valid_y)
prediction_f1_score = f1_score(valid_y, predicted_valid, average='weighted')
prediction_recall = recall_score(valid_y, predicted_valid, average='weighted')
print('Model NaiveBayes: ', 'Accuracy: ', prediction_acc,
      'F1: ', prediction_f1_score, 'Recall: ', prediction_recall)

classifier = RandomForestClassifier(n_estimators=1000, random_state=0)
classifier.fit(train_X, train_y)
predicted_valid = classifier.predict(valid_X)
prediction_acc = np.mean(predicted_valid == valid_y)
prediction_f1_score = f1_score(valid_y, predicted_valid, average='weighted')
prediction_recall = recall_score(valid_y, predicted_valid, average='weighted')
print('Model Random Forrest Classifier: ','Accuracy: ', prediction_acc,
      'F1: ', prediction_f1_score, 'Recall: ', prediction_recall)


import pandas as pd
from imblearn.under_sampling import  RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

df_review = pd.read_csv('IMDB Dataset.csv')

df_positive = df_review[df_review['sentiment']=='positive'][:9000]
df_negative = df_review[df_review['sentiment']=='negative'][:1000]
df_review_imb = pd.concat([df_positive, df_negative])

rus = RandomUnderSampler(random_state=0)
df_review_bal, df_review_bal['sentiment']=rus.fit_resample(df_review_imb[['review']], df_review_imb['sentiment'])

train, test = train_test_split(df_review_bal, test_size=0.33, random_state=42)

train_x, train_y = train['review'], train['sentiment']
test_x, test_y = test['review'], test['sentiment']

tfidf = TfidfVectorizer(stop_words='english')
train_x_vector = tfidf.fit_transform(train_x)

pd.DataFrame.sparse.from_spmatrix(train_x_vector, index=train_x.index, columns=tfidf.get_feature_names())
test_x_vector = tfidf.transform(test_x)

svc = SVC(kernel='linear')
svc.fit(train_x_vector, train_y)

dec_tree = DecisionTreeClassifier()
dec_tree.fit(train_x_vector, train_y)

gnb = GaussianNB()
gnb.fit(train_x_vector.toarray(), train_y)

log_reg = LogisticRegression()
log_reg.fit(train_x_vector, train_y)

# svc.score('Test samples', 'True labels')
print("SVM:" + str(svc.score(test_x_vector, test_y)))
print("Decision Trees:" + str(dec_tree.score(test_x_vector, test_y)))
print("Naive Bayes Classifier:" + str(gnb.score(test_x_vector.toarray(), test_y)))
print("Logistic Regression: " + str(log_reg.score(test_x_vector, test_y)))

print(classification_report(test_y, svc.predict(test_x_vector), labels=['positive', 'negative']))

conf_mat = confusion_matrix(test_y, svc.predict(test_x_vector), labels=['positive', 'negative'])







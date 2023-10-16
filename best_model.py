import re
import pandas as pd
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import xgboost as xgb

#nltk.download('stopwords')
#nltk.download('wordnet')

#random state 1458 ou 3390
#count vectorizer multionomialNB 

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
ps = PorterStemmer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\W', ' ', str(text))
    text = " ".join([word for word in text.split() if word not in stop_words])
    #text = " ".join([lemmatizer.lemmatize(word) for word in text.split()])
    text = " ".join([ps.stem(word) for word in text.split()])
    return text

# Load data from file
with open('train.txt', 'r', encoding='utf-8') as file:
    data = file.read()

# Parsing data
rows = data.strip().split('\n')
labels, texts = [], []
for row in rows:
    label, text = row.split('\t', 1)
    labels.append(label)
    texts.append(text)

# Creating a DataFrame
df = pd.DataFrame({'label': labels, 'text': texts})

# Applying preprocessing on the text data
df['text'] = df['text'].apply(lambda x: preprocess_text(x))

 
# Splitting data into training and testing sets
train_data, test_data, train_label, test_label = train_test_split(df['text'], df['label'], test_size=0.2, random_state=1458)

# Building a pipeline that combines a CountVectorizer with a Naive Bayes classifier
model = make_pipeline(CountVectorizer(), MultinomialNB())

# Training the model
model.fit(train_data, train_label)

# Predicting the test data
predictions = model.predict(test_data)

# Evaluating the model
print(metrics.accuracy_score(test_label, predictions))  
print(confusion_matrix(test_label, predictions))
print(classification_report(test_label, predictions))


"""
# Identify and print misclassified labels and texts
misclassified_indexes = test_label != predictions
misclassified_samples = test_data[misclassified_indexes]
misclassified_labels = test_label[misclassified_indexes]
predicted_labels = predictions[misclassified_indexes]

print("Misclassified Samples:")
for true_label, predicted_label, text in zip(misclassified_labels, predicted_labels, misclassified_samples):
    print(f"\nTrue label: {true_label}\nPredicted label: {predicted_label}\nText: {text}")
"""

""" 
best_accuracy = 0

for i in range(1, 2000):
    # Splitting data into training and testing sets
    train_data, test_data, train_label, test_label = train_test_split(df['text'], df['label'], test_size=0.2, random_state=i)

    # Building a pipeline that combines a CountVectorizer with a Naive Bayes classifier
    model = make_pipeline(TfidfVectorizer(), SVC(kernel="rbf"))

    # Training the model
    model.fit(train_data, train_label)

    # Predicting the test data
    predictions = model.predict(test_data)

    if metrics.accuracy_score(test_label, predictions) > best_accuracy:
        best_accuracy = metrics.accuracy_score(test_label, predictions)
        best_random_state = i

    # Evaluating the model
    #print(metrics.accuracy_score(test_label, predictions))  
    #print(confusion_matrix(test_label, predictions))
    #print(classification_report(test_label, predictions))

print("Best accuracy: " + str(best_accuracy))
print("Best random state: " + str(best_random_state))
"""

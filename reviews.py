import re
import pandas as pd
from sklearn.model_selection import train_test_split
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score


lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    text = text.replace('...', '«three_dots»')
    text = text.replace('!', '«exclamation_mark»')
    text = text.replace('?', '«question_mark»')
    text = text.replace(':)', '«emoji»')
    text = text.replace(':(', '«emoji»')
   
    text = text.replace("am", ' ')
    text = text.replace("pm", ' ')

    # Tokenize the text, lemmatize each token, then join back into a string
    text = " ".join([lemmatizer.lemmatize(word) for word in text.split()])

    # Handle punctuation and numbers
    text = re.sub(r'(\d)[a-zA-Z]+', r'\1', text) #Remove text after numbers: 7th -> 7
    text = re.sub(r'[^\w\s«exclamation_mark»«question_mark»«three_dots»«emoji»»]', ' ', str(text)) 
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

"""
########## Using GridSearchCV to find the best parameters - Takes a long time ##########
param_grid = {
    'vect__ngram_range': [(1, 1), (1, 2), (2, 2)],
    'vect__max_df': [0.5, 0.75, 1.0],
    'vect__min_df': [1, 2, 3],
    'clf__C': [0.1, 1, 10],
    'clf__kernel': ['linear', 'rbf'],
    'clf__gamma': ['scale', 'auto']
}

pipeline = Pipeline([
    ('vect', TfidfVectorizer()),
    ('clf', SVC())
])

grid_search = GridSearchCV(pipeline, param_grid, cv=10, scoring='accuracy', verbose=1, n_jobs=-1)
grid_search.fit(df['text'], df['label'])

best_params = grid_search.best_params_
best_estimator = grid_search.best_estimator_
print(best_params)
print(best_estimator)

New best
{'clf__C': 10, 'clf__gamma': 'scale', 'clf__kernel': 'linear', 'vect__max_df': 1.0, 'vect__min_df': 1, 'vect__ngram_range': (1, 2)}
Pipeline(steps=[('vect', TfidfVectorizer(ngram_range=(1, 2))),
                ('clf', SVC(C=10, kernel='linear'))])

# Configure TfidfVectorizer and SVC with the best parameters
tfidf = TfidfVectorizer(max_df=0.25, ngram_range=(1,2))
svc = SVC(C=10, kernel='sigmoid')
"""


# Build the final pipeline using the best parameters
final_pipeline = Pipeline([('vect', TfidfVectorizer(ngram_range=(1, 2), max_df=1.0, min_df=1)),
                            ('clf', SVC(C=10, gamma='scale', kernel='linear'))])


""" 
########## Using cross validation to find the mean accuracy ##########

# Cross-validation on final pipeline
final_scores = cross_val_score(final_pipeline, df['text'], df['label'], cv=10, scoring='accuracy')

# Calculate mean accuracy
mean_accuracy = final_scores.mean()

# Print results
#print("Best Parameters: ", best_params)
print("Final Cross-Validation Scores: ", final_scores)
print("Mean Accuracy: ", mean_accuracy)
"""


train_data, test_data, train_label, test_label = train_test_split(df['text'], df['label'], test_size=0.2, random_state=1)

final_pipeline.fit(train_data, train_label)

predictions = final_pipeline.predict(test_data)

accuracy = accuracy_score(test_label, predictions)


""" 
########## Miss Predictions ##########
# Create a DataFrame with actual and predicted labels
results_df = pd.DataFrame({
    'Actual': test_label,
    'Predicted': predictions
})

# Find mispredictions
mispredictions = results_df[results_df['Actual'] != results_df['Predicted']]

# Merge the mispredictions DataFrame with the test_data DataFrame based on the index
mispredictions_with_text = mispredictions.merge(test_data, left_index=True, right_index=True)

# Rename the columns for clarity
mispredictions_with_text.columns = ['Actual', 'Predicted', 'Text']

# Now mispredictions_with_text contains the actual labels, predicted labels, and the text
# Save this DataFrame to a file if desired
mispredictions_with_text.to_csv('mispredictions_with_text.csv', index=False)

# Output mispredictions to the console
print(mispredictions_with_text)
"""


########### Testing ###########

with open('test_just_reviews.txt', 'r', encoding='utf-8') as file:
    new_data = file.read().strip().split('\n')

# Create a DataFrame for the new data
new_df = pd.DataFrame({'text': new_data})

# Apply the same preprocessing steps to the new data
new_df['text'] = new_df['text'].apply(lambda x: preprocess_text(x))

# Use the trained model to make predictions
new_predictions = final_pipeline.predict(new_df['text'])

# Save the predicted labels to a text file
with open('results.txt', 'w', encoding='utf-8') as file:
    for label in new_predictions:
        file.write(f"{label}\n")
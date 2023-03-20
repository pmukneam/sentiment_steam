# Importing libraries
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import f1_score

## Importing data

# Importing Steam Review
original_data = pd.read_csv("dataset/dataset.csv")
steam_review = original_data.head(1000000)

# Selecting only the review and score column
steam_review = steam_review[["review_text", "review_score"]]

# Droping NULL/NA rows
steam_review = steam_review.dropna()

# Droping duplicate
steam_review = steam_review.drop_duplicates()

## Data Cleaning

# Removing trailing space and make all reviews lowercase
steam_review['review_text'] = [review.strip().lower() for review in steam_review['review_text']]

# Removing special characters
steam_review['review_text'] = steam_review['review_text'].replace(r"[^a-zA-Z\d\_\+\-\'\.\/\s]+", ' ', regex = True)
steam_review['review_text'] = steam_review['review_text'].replace(["./ ", "' ", " '"], " ", regex = True)

# Importing stopwords in English
stopwords = stopwords.words('english')

# Creating stopwords regex
pat = r'\b(?:{})\b'.format('|'.join(stopwords))

# Removing stopwords
steam_review['review_text'] = steam_review['review_text'].str.replace(pat, '')
steam_review['review_text'] = steam_review['review_text'].str.replace(r'\s+', ' ')

## Building Models

# Spliting the reviews into training and testing dataset
data_frame = steam_review
data_frame, test_data = train_test_split(data_frame, test_size = 0.20)
train_data, dev_data = train_test_split(data_frame, test_size = 0.20)

# Transforming each reviews into vectors using TF-IDF
# Limiting the number of words per review to 1000 words
tfidf = TfidfVectorizer(max_features=1000)

# Training
x_train = tfidf.fit_transform(train_data['review_text']).toarray()
y_train = np.asarray(train_data['review_score'])

# Local testing
x_dev = tfidf.transform(dev_data['review_text']).toarray()
y_dev = np.asarray(dev_data['review_score'])

## Modeling

# Naive Bayes with different alpha for Laplacian smoothing
alpha = [0.001, 0.01, .1, 0.5, 1, 2, 3]
multinomial_nb = {}
mse_multinomial_nb = {}
accuracy_multinomial_nb = {}
f1_multinomial_nb = {}
y_dev_pred_multinomial_nb = {}

print('Naive Bayes with Laplacian Smoothing:\n')

# Loop through each alpha
for a in alpha:
    multinomial_nb[a] = MultinomialNB(alpha=a)
    multinomial_nb[a].fit(x_train, y_train)

    y_dev_pred_multinomial_nb[a] = (multinomial_nb[a].predict(x_dev))

    # Calculate the Mean Squared Error, Accuracy, and F-1 Score
    mse_multinomial_nb[a] = mean_squared_error(y_dev, y_dev_pred_multinomial_nb[a])
    accuracy_multinomial_nb[a] = accuracy_score(y_dev, y_dev_pred_multinomial_nb[a])*100
    f1_multinomial_nb[a] = f1_score(y_dev, y_dev_pred_multinomial_nb[a])

    # Print the Mean Squared Error, Accuracy, and F-1 Score
    print(f'Mean Squared Error = {mse_multinomial_nb[a]} for alpha = {a}')
    print(f'Accuracy = {accuracy_multinomial_nb[a]} for alpha = {a}')
    print(f'F-1 Score = {f1_multinomial_nb[a]} for alpha = {a}\n')

# Store everything inside an array for easier acess
mse = {}
accuracy = {}
f1 = {}
model = {}
model["MNB"] = multinomial_nb
mse["MNB"] = mse_multinomial_nb
accuracy["MNB"] = accuracy_multinomial_nb
f1["MNB"] = f1_multinomial_nb

# Support Vector Machine
C = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
svm = {}
mse_svm = {}
accuracy_svm = {}
f1_svm = {}
y_dev_pred_svm = {}

print('Linear Support Vector Machine\n')

# Loop through each alpha
for c in C:
    # Suppose there is a linear relationship between reviews and its score
    svm[c] = LinearSVC(C = c, dual = False)
    svm[c].fit(x_train, y_train)

    y_dev_pred_svm[c] = (svm[c].predict(x_dev))

    # Calculate the Mean Squared Error, Accuracy, and F-1 Score
    mse_svm[c] = mean_squared_error(y_dev, y_dev_pred_svm[c])
    accuracy_svm[c] = accuracy_score(y_dev, y_dev_pred_svm[c])*100
    f1_svm[c] = f1_score(y_dev, y_dev_pred_svm[c])

    # Print the Mean Squared Error, Accuracy, and F-1 Score
    print(f'Mean Squared Error = {mse_svm[c]} for C = {c}')
    print(f'Accuracy = {accuracy_svm[c]} for C = {c}')
    print(f'F-1 Score = {f1_svm[c]} for C = {c}\n')

## Testing

# Since SVM at c = 100 seems to yields a high accuracy
# Let test with our testing data set

# Transforming each reviews into vectors using TF-IDF
x_test = tfidf.transform(test_data['review_text']).toarray()
y_test = np.asarray(test_data['review_score'])

# Setting SVM
c=100
y_test_pred = (svm[c].predict(x_test))

# Calculate the Mean Squared Error, Accuracy, and F-1 Score

mse_test = mean_squared_error(y_test, y_test_pred)
accuracy_test = accuracy_score(y_test, y_test_pred)*100
f1_test = f1_score(y_test, y_test_pred)

print(f'Testing SVM with c = {c} on the testing dataset:')
print(f'Mean Squared Error test = {mse_test}')
print(f'Accuracy test = {accuracy_test}')
print(f'F-1 Score test = {f1_test}\n')

## Store the models
model['SVM'] = svm
mse['SVM'] = mse_svm
accuracy['SVM'] = accuracy_svm
f1['SVM'] = f1_svm

"""
svmmnb: This function takes an arrays of a review, then returns its sentiment
input:
        x: an array of a review (string)
output:
        numberical sentiment (pos=1, neg=-1)
"""
def svmmnb(x):
    x_array = tfidf.transform(x).toarray()
    y_pred_dict = {}

    # Using Naive Bayes
    for a in alpha:
        y_pred = model['MNB'][a].predict(x_array)[0]
        if y_pred in y_pred_dict:
            y_pred_dict[y_pred] += (accuracy['MNB'][a])
        else:
            y_pred_dict[y_pred] = (accuracy['MNB'][a])

    # Using SVM
    for c in C:
        y_pred = model['SVM'][c].predict(x_array)[0]
        if y_pred in y_pred_dict:
            y_pred_dict[y_pred] += (accuracy['SVM'][c])
        else:
            y_pred_dict[y_pred] = (accuracy['SVM'][c])

    # Returning the result from the most accurate ones
    inverse = [(value, key) for key, value in y_pred_dict.items()]
    return max(inverse)[1]

r1_score = svmmnb(['This game is so bad. It ruins my life. I hate it but I cant stop playing it tho still a terrible game. One of the worst on the market.'])
print('Review 1: This game is so bad. It ruins my life. I hate it but I cant stop playing it tho. Still a terrible game.')
print(f'score: {r1_score}\n')

r2_score = svmmnb(['One of the best game I ever play.'])
print('Review 2: One of the best game I ever play.')
print(f'score: {r2_score}')

# EDA Packages
import pandas as pd
import numpy as np
import random


# Machine Learning Packages
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score

# Load Url Data 
urls_data = pd.read_csv("urldata.csv")


type(urls_data)

urls_data.head()

def makeTokens(f):
    tkns_BySlash = str(f.encode('utf-8')).split('/')	# make tokens after splitting by slash
    total_Tokens = []
    for i in tkns_BySlash:
        tokens = str(i).split('-')	# make tokens after splitting by dash
        tkns_ByDot = []
        for j in range(0,len(tokens)):
            temp_Tokens = str(tokens[j]).split('.')	# make tokens after splitting by dot
            tkns_ByDot = tkns_ByDot + temp_Tokens
        total_Tokens = total_Tokens + tokens + tkns_ByDot
    total_Tokens = list(set(total_Tokens))	#remove redundant tokens
    if 'com' in total_Tokens:
        total_Tokens.remove('com')	#removing .com since it occurs a lot of times and it should not be included in our features
    return total_Tokens

# Labels
y = urls_data["label"]


# Features
url_list = urls_data["url"]

# Using Default Tokenizer
#vectorizer = TfidfVectorizer()

# Using Custom Tokenizer
vectorizer = TfidfVectorizer(tokenizer=makeTokens)

# Store vectors into X variable as Our XFeatures
X = vectorizer.fit_transform(url_list)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.001, random_state=42)

# instantiate learning model (k = 3)
knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1, leaf_size=20)
print('Generated Learning model!')
# fitting the model
knn.fit(X_train, y_train)
print('Fit the model!')



X_predict = ["google.com/search=jcharistech",
"google.com/search=faizanahmad",
"pakistanifacebookforever.com/getpassword.php/", 
"www.radsport-voggel.de/wp-admin/includes/log.exe", 
"ahrenhei.without-transfer.ru/nethost.exe ",
"www.itidea.it/centroesteticosothys/img/_notes/gum.exe",
"paypal-manager-loesung.net/konflikt/66211165125/",
"stackoverflow.com/questions/8551735/how-do-i-run-python-code-from-sublime-text-2",
"146.71.84.110"]
X_predict = vectorizer.transform(X_predict)


# predict the response
New_predict = knn.predict(X_predict)

print('Predicting Response!')
print(New_predict)
# evaluate accuracy

probabilities = knn.predict_proba(X_predict)
print(probabilities)

y_true = [1, 1, 0, 0, 0, 0, 1, 1, 0]

# Results Section
score = knn.score(X_test, y_test)
print('Accuracy: ', score)
probabilities_array = np.array((probabilities[:,1]))
print('Probabilities: ', probabilities_array)
avg_prec = average_precision_score(np.array(y_true), probabilities_array)
print('Precision: ', avg_prec)
recall = recall_score(y_true, New_predict, average='binary')
print('Recall: ', recall)


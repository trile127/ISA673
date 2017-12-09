# EDA Packages
import pandas
import numpy
import random

# Machine Learning Algorithm Packages
from sklearn import linear_model                        # Linerar Regrssion
from sklearn.linear_model import LogisticRegression     # Logistic Regression
from sklearn import tree                                # Decision Tree
from sklearn import svm                                 # Support Vector Machine (SVM)
from sklearn.naive_bayes import GaussianNB              # Naive Bayes
from sklearn.neighbors import KNeighborsClassifier      # K-Nearest Neighbors (KNN)
from sklearn.cluster import KMeans                      # K-Means
from sklearn.ensemble import RandomForestClassifier     # Random Forest

# Mahine Learning Scoring Packages
from sklearn.metrics import accuracy_score              # Accuracy
from sklearn.metrics import average_precision_score     # Precision
from sklearn.metrics import recall_score                # Recall

# Machine Learning Prep Packages
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict

# Machine Learning Plot Packages
from matplotlib import pyplot as plt
import seaborn as sns

# Load Url Data
urls_data = pandas.read_csv("url_0.txt")

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
vectorizer = TfidfVectorizer(tokenizer=makeTokens)

# Store vectors into X variable as Our XFeatures
x = vectorizer.fit_transform(url_list)
x_trn, x_tst, y_trnLabel, y_tstLabel = train_test_split(x, y, test_size=0.3, random_state=42)

def linearregression():
	print '\n### Running Linear Regression Algorithm\n'

	# Create Linear Regression classifier object model
	model = linear_model.LinearRegression()

	# Train the model using the training sets and check score
	model.fit(x_trn, y_trnLabel)
	print '\tTraining score: ', model.score(x_trn, y_trnLabel)

	#Predict Output
	predicted= model.predict(x_tst)
	probabilities_threshold = (predicted > .5).astype(numpy.int)

	# Test the model using the testing sets and check score
	print '\tTesting score: ', model.score(x_tst, predicted)
	print '\tAccuracy score: ', accuracy_score(y_tstLabel, probabilities_threshold)
	avg_prec = average_precision_score(numpy.array(y_tstLabel), probabilities_threshold)
	print('\tPrecision Score: ' + str(avg_prec))
	recall = recall_score(y_tstLabel, probabilities_threshold, average='micro')
	print('\tRecall Score: ' +  str(recall))


def logisticregression():
	print '\n### Running Logistic Regression Algorithm\n'

	# Create Logistic Regression classifier object model
	model = LogisticRegression()

	# Train the model using the training sets and check score
	model.fit(x_trn, y_trnLabel)
	print '\tTraining score: ', model.score(x_trn, y_trnLabel)

	#Predict Output
	predicted= model.predict(x_tst)
	# Test the model using the testing sets and check score
	print '\tTesting score: ', model.score(x_tst, predicted)
	print '\tAccuracy score: ', accuracy_score(y_tstLabel, predicted)
	avg_prec = average_precision_score(numpy.array(y_tstLabel), predicted)
	print('\tPrecision Score: ' + str(avg_prec))
	recall = recall_score(y_tstLabel, predicted, average='micro')
	print('\tRecall Score: ' +  str(recall))

def decisiontree():
	print '\n### Running Decision Tree Algorithm\n'

	# Create Decision Tree classifier object model
	model = tree.DecisionTreeClassifier(criterion='gini') # for classification, here you can change the algorithm as gini or entropy (information gain) by default it is gini  
	# model = tree.DecisionTreeRegressor() for regression

	# Train the model using the training sets and check score
	model.fit(x_trn, y_trnLabel)
	print '\tTraining score: ', model.score(x_trn, y_trnLabel)

	#Predict Output
	predicted= model.predict(x_tst)
	
	probabilities = model.predict_proba(x_tst)
	probabilities_array = numpy.array((probabilities[:,1]))
	
	# Test the model using the testing sets and check score
	print '\tTesting score: ', model.score(x_tst, predicted)
	print '\tAccuracy score: ', accuracy_score(y_tstLabel, probabilities_array)
	avg_prec = average_precision_score(numpy.array(y_tstLabel), probabilities_array)
	print('\tPrecision Score: ' + str(avg_prec))
	recall = recall_score(y_tstLabel, probabilities_array, average='micro')
	print('\tRecall Score: ' +  str(recall))

def supportvectormachine():
	print '\n### Running Support Vector Machine Algorithm\n'

	# Create Support Vector Machine classifier object model
	model = svm.SVC(probability=True) # there is various option associated with it, this is simple for classification.

	# Train the model using the training sets and check score
	model.fit(x_trn, y_trnLabel)
	print '\tTraining score: ', model.score(x_trn, y_trnLabel)

	#Predict Output
	predicted= model.predict(x_tst)

	probabilities = model.predict_proba(x_tst)
	probabilities_array = numpy.array((probabilities[:,1]))
	probabilities_threshold = (probabilities_array > .5).astype(numpy.int)
	
	# Test the model using the testing sets and check score
	print '\tTesting score: ', model.score(x_tst, probabilities_threshold)
	print '\tAccuracy score: ', accuracy_score(y_tstLabel, probabilities_threshold)
	avg_prec = average_precision_score(numpy.array(y_tstLabel), probabilities_threshold)
	print('\tPrecision Score: ' + str(avg_prec))
	recall = recall_score(y_tstLabel, probabilities_threshold, average='micro')
	print('\tRecall Score: ' +  str(recall))

def naivebayes():
	print '\n### Running Naive Bayes Algorithm\n'

	# Create Naive Bayes classifier object model
	model = GaussianNB()

        x1 = x.toarray()	# Convert to a dense numpy array.
	
	# Store vectors into X variable as Our XFeatures
	x1_trn, x1_tst, y1_trnLabel, y1_tstLabel = train_test_split(x1, y, test_size=0.2, random_state=42)

	# Train the model using the training sets and check score
	model.fit(x1_trn, y1_trnLabel)
	print '\tTraining score: ', model.score(x1_trn, y1_trnLabel)

	#Predict Output
	predicted= model.predict(x1_tst)

	# Test the model using the testing sets and check score
	print '\tTesting score: ', model.score(x1_tst, predicted)
	print '\tAccuracy score: ', accuracy_score(y1_tstLabel, predicted)
	avg_prec = average_precision_score(numpy.array(y1_tstLabel), predicted)
	print('\tPrecision Score: ' + str(avg_prec))
	recall = recall_score(y1_tstLabel, predicted, average='micro')
	print('\tRecall Score: ' +  str(recall))

def knearestneighbors():
	print '\n### Running K-Nearest Neighbors Algorithm\n'

	# Create K-Nearest Neighbors classifier object model
	model = KNeighborsClassifier(n_neighbors=5, leaf_size=25) # default value for n_neighbors is 5

	# Train the model using the training sets and check score
	model.fit(x_trn, y_trnLabel)
	print '\tTraining score: ', model.score(x_trn, y_trnLabel)

	#Predict Output
	predicted= model.predict(x_tst)
	probabilities = model.predict_proba(x_tst)
	#print(predicted)
	# Test the model using the testing sets and check score
	print '\tTesting score: ', model.score(x_tst, y_tstLabel)
	probabilities_array = numpy.array((probabilities[:,1]))
	print '\tAccuracy score: ', accuracy_score(y_tstLabel, predicted)
	#print('Probabilities Score: ' + str(probabilities_array))
	avg_prec = average_precision_score(numpy.array(y_tstLabel), probabilities_array)
	print('\tPrecision Score: ' + str(avg_prec))
	recall = recall_score(y_tstLabel, predicted, average='micro')
	print('\tRecall Score: ' +  str(recall))

	
	

def kmeans():
	print '\n### Running K-Means Algorithm\n'

	# Create K-Means classifier object model
	model = KMeans(n_clusters=8, random_state=0)

	# Train the model using the training sets and check score
	model.fit(x_trn)
	print '\tTraining score: ', model.score(x_trn, y_trnLabel)

	#Predict Output
	predicted= model.predict(x_tst)
	
	# Test the model using the testing sets and check score
	print '\tTesting score: ', model.score(x_tst, predicted)
	print '\tAccuracy score: ', accuracy_score(y_tstLabel, predicted)
	avg_prec = average_precision_score(numpy.array(y_tstLabel), predicted)
	print('\tPrecision Score: ' + str(avg_prec))
	recall = recall_score(y_tstLabel, predicted, average='micro')
	print('\tRecall Score: ' +  str(recall))

def randomforest():
	print '\n### Running Random Forest Algorithm\n'

	# Create Random Forest classifier object model
	model = RandomForestClassifier()

	# Train the model using the training sets and check score
	model.fit(x_trn, y_trnLabel)
	print '\tTraining score: ', model.score(x_trn, y_trnLabel)

		#Predict Output
	predicted= model.predict(x_tst)

	# Test the model using the testing sets and check score
	print '\tTesting score: ', model.score(x_tst, y_tstLabel)
	print '\tAccuracy score: ', accuracy_score(y_tstLabel, predicted)
	avg_prec = average_precision_score(numpy.array(y_tstLabel), predicted)
	print('\tPrecision Score: ' + str(avg_prec))
	recall = recall_score(y_tstLabel, predicted, average='micro')
	print('\tRecall Score: ' +  str(recall))

# Run the machine learning algorithms
linearregression()
logisticregression()
decisiontree()
supportvectormachine()
naivebayes()
knearestneighbors()
kmeans()
randomforest()
print '\n'	#  Add space to read results

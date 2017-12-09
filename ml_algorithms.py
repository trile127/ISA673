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

# Machine Learning save data
from sklearn.externals import joblib

# Load Url Data
urls_data = pandas.read_csv("url_0.txt")


# Prediction Lists Results for each algorithm
linearReg_Pred_List = []
logisticReg_Pred_List = []
decisionTree_Pred_List = []
supportVectorMachine_Pred_List = []
naiveBayes_Pred_List = []
knearestNeighbors_Pred_List = []
kMeans_Pred_List = []
randomForest_Pred_List = []


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
x_trn, x_tst, y_trnLabel, y_tstLabel = train_test_split(x, y, test_size=0.1, random_state=42)

def linearregression(x_predict_url):
	if (x_predict_url == None):
		print '\n### Running Linear Regression Algorithm\n'

		# Create Linear Regression classifier object model
		model = linear_model.LinearRegression()

		# Train the model using the training sets and check score
		model.fit(x_trn, y_trnLabel)
		print '\tTraining score: ', model.score(x_trn, y_trnLabel)

		#Predict Output
		predicted= model.predict(x_tst)
		probabilities_threshold = (predicted > .6).astype(numpy.int)

		# Test the model using the testing sets and check score
		print '\tTesting score: ', model.score(x_tst, predicted)
		print '\tAccuracy score: ', accuracy_score(y_tstLabel, probabilities_threshold)
		avg_prec = average_precision_score(numpy.array(y_tstLabel), probabilities_threshold)
		print('\tPrecision Score: ' + str(avg_prec))
		recall = recall_score(y_tstLabel, probabilities_threshold, average='micro')
		print('\tRecall Score: ' +  str(recall))
		
		filename = '/tmp/linearregression_classifier.joblib.pkl'
		joblib.dump(model, filename, compress=9)
		
	else:
		global linearReg_Pred_List

		filename = '/tmp/linearregression_classifier.joblib.pkl'
		model = joblib.load(filename)
		s = pandas.Series(x_predict_url)
		X_predict = vectorizer.transform(s)
		New_predict = model.predict(X_predict)
		New_predict_threshold = (New_predict > .6).astype(numpy.int)
		linearReg_Pred_List.append(New_predict_threshold)
		if (New_predict_threshold == 0):
			print(x_predict_url + " is benign!!")
		else:
			print(x_predict_url + " is malicious!!")
		


def logisticregression(x_predict_url):
	if (x_predict_url == None):
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
		filename = '/tmp/logisticregression_classifier.joblib.pkl'
		joblib.dump(model, filename, compress=9)
		
	else:
		global logisticReg_Pred_List
		filename = '/tmp/logisticregression_classifier.joblib.pkl'
		model = joblib.load(filename)
		s = pandas.Series(x_predict_url)
		X_predict = vectorizer.transform(s)
		New_predict = model.predict(X_predict)
		logisticReg_Pred_List.append(New_predict)
		if (New_predict == 0):
			print(x_predict_url + " is benign!!")
		else:
			print(x_predict_url + " is malicious!!")

def decisiontree(x_predict_url):
	if (x_predict_url == None):
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
		filename = '/tmp/decisiontree_classifier.joblib.pkl'
		joblib.dump(model, filename, compress=9)
		
	else:
		global decisionTree_Pred_List
		filename = '/tmp/decisiontree_classifier.joblib.pkl'
		model = joblib.load(filename)
		s = pandas.Series(x_predict_url)
		X_predict = vectorizer.transform(s)
		New_predict = model.predict(X_predict)
		decisionTree_Pred_List.append(New_predict)
		if (New_predict == 0):
			print(x_predict_url + " is benign!!")
		else:
			print(x_predict_url + " is malicious!!")

def supportvectormachine(x_predict_url):
	if (x_predict_url == None):
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
		
		filename = '/tmp/supportvectormachine_classifier.joblib.pkl'
		joblib.dump(model, filename, compress=9)
		
	else:
		global supportVectorMachine_Pred_List
		filename = '/tmp/supportvectormachine_classifier.joblib.pkl'
		model = joblib.load(filename)
		s = pandas.Series(x_predict_url)
		X_predict = vectorizer.transform(s)
		New_predict = model.predict(X_predict)

		New_predict_probabilities = model.predict_proba(X_predict)
		New_predict_probabilities_array = numpy.array((New_predict_probabilities[:,1]))
		New_predict_threshold = (New_predict_probabilities_array > .5).astype(numpy.int)
		supportVectorMachine_Pred_List.append(New_predict_threshold)
		if (New_predict_threshold == 0):
			print(x_predict_url + " is benign!!")
		else:
			print(x_predict_url + " is malicious!!")

def naivebayes(x_predict_url):
	if (x_predict_url == None):
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

		filename = '/tmp/naivebayes_classifier.joblib.pkl'
		joblib.dump(model, filename, compress=9)
		
	else:
		global naiveBayes_Pred_List
		filename = '/tmp/naivebayes_classifier.joblib.pkl'
		model = joblib.load(filename)
		s = pandas.Series(x_predict_url)
		X_predict = vectorizer.transform(s).toarray()
		New_predict = model.predict(X_predict)
		naiveBayes_Pred_List.append(New_predict)
		if (New_predict == 0):
			print(x_predict_url + " is benign!!")
		else:
			print(x_predict_url + " is malicious!!")

def knearestneighbors(x_predict_url):
	if (x_predict_url == None):
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
		avg_prec = average_precision_score(numpy.array(y_tstLabel), predicted)
		print('\tPrecision Score: ' + str(avg_prec))
		recall = recall_score(y_tstLabel, predicted, average='micro')
		print('\tRecall Score: ' +  str(recall))
		
		filename = '/tmp/knearestneighbors_classifier.joblib.pkl'
		joblib.dump(model, filename, compress=9)
		
	else:
		global knearestNeighbors_Pred_List
		filename = '/tmp/knearestneighbors_classifier.joblib.pkl'
		model = joblib.load(filename)
		s = pandas.Series(x_predict_url)
		X_predict = vectorizer.transform(s)
		New_predict = model.predict(X_predict)
		knearestNeighbors_Pred_List.append(New_predict)

		if (New_predict == 0):
			print(x_predict_url + " is benign!!")
		else:
			print(x_predict_url + " is malicious!!")
	
	

def kmeans(x_predict_url):
	if (x_predict_url == None):
		print '\n### Running K-Means Algorithm\n'

		# Create K-Means classifier object model
		model = KMeans(n_clusters=2, random_state=0)

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
	
		filename = '/tmp/kmeans_classifier.joblib.pkl'
		joblib.dump(model, filename, compress=9)
		
	else:
		global kMeans_Pred_List
		filename = '/tmp/kmeans_classifier.joblib.pkl'
		model = joblib.load(filename)
		model = joblib.load(filename)
		s = pandas.Series(x_predict_url)
		X_predict = vectorizer.transform(s)
		New_predict = model.predict(X_predict)
		kMeans_Pred_List.append(New_predict)
		if (New_predict == 0):
			print(x_predict_url + " is benign!!")
		else:
			print(x_predict_url + " is malicious!!")

def randomforest(x_predict_url):
	if (x_predict_url == None):
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
	
		filename = '/tmp/randomforest_classifier.joblib.pkl'
		joblib.dump(model, filename, compress=9)
		
	else:
		global randomForest_Pred_List
		filename = '/tmp/randomforest_classifier.joblib.pkl'
		model = joblib.load(filename)
		s = pandas.Series(x_predict_url)
		X_predict = vectorizer.transform(s)
		New_predict = model.predict(X_predict)
		randomForest_Pred_List.append(New_predict)
		if (New_predict == 0):
			print(x_predict_url + " is benign!!")
		else:
			print(x_predict_url + " is malicious!!")

			

			
# Run the machine learning algorithms
linearregression(None)
logisticregression(None)
decisiontree(None)
supportvectormachine(None)
naivebayes(None)
knearestneighbors(None)
kmeans(None)
randomforest(None)
print '\n'	#  Add space to read results



X_predictions = ["google.com/search=jcharistech",
"google.com/search=faizanahmad",
"pakistanifacebookforever.com/getpassword.php/", 
"www.radsport-voggel.de/wp-admin/includes/log.exe", 
"ahrenhei.without-transfer.ru/nethost.exe ",
"www.itidea.it/centroesteticosothys/img/_notes/gum.exe",
"paypal-manager-loesung.net/konflikt/66211165125/",
"stackoverflow.com/questions/8551735/how-do-i-run-python-code-from-sublime-text-2",
"146.71.84.110"]

for URL in X_predictions:
	linearregression(URL)
	logisticregression(URL)
	decisiontree(URL)
	supportvectormachine(URL)
	naivebayes(URL)
	knearestneighbors(URL)
	kmeans(URL)
	randomforest(URL)


print("\nPrediction Lists: ")

all_pred_list = []
all_pred_list.append(linearReg_Pred_List)
all_pred_list.append(logisticReg_Pred_List)
all_pred_list.append(decisionTree_Pred_List)
all_pred_list.append(supportVectorMachine_Pred_List)
all_pred_list.append(naiveBayes_Pred_List)
all_pred_list.append(knearestNeighbors_Pred_List)
all_pred_list.append(randomForest_Pred_List)

actual_Labels = [0, 0, 1, 1, 1, 1, 0, 0, 1]

# Print results
def print_all_results(results_list):
	counter = 0
	alg_list = ['Linear Regression', 'Logistic Regression', 'Decision Tree', 'SVM', 'Naive Bayes', 'KNN', 'Kmeans', 'Random Forest']
	for list in results_list:
		print(alg_list[counter])
		print("Actuals: ", actual_Labels)
		print '\tAccuracy score: ', accuracy_score(actual_Labels, list)
		avg_prec = average_precision_score(numpy.array(actual_Labels), list)
		print('\tPrecision Score: ' + str(avg_prec))
		recall = recall_score(actual_Labels, list, average='micro')
		print('\tRecall Score: ' +  str(recall))
		counter = counter + 1
		
print_all_results(all_pred_list)
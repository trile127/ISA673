# EDA Packages
import pandas
import numpy
import random
import classifier                                       # Feature Extraction

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

# Machine Learning Plot Packages
from matplotlib import pyplot as plt
import seaborn as sns

# Machine Learning save data
from sklearn.externals import joblib

# Prediction Lists Results for each algorithm
linearReg_Pred_List = []
logisticReg_Pred_List = []
decisionTree_Pred_List = []
supportVectorMachine_Pred_List = []
naiveBayes_Pred_List = []
knearestNeighbors_Pred_List = []
kMeans_Pred_List = []
randomForest_Pred_List = []


# Load Url Data
urls_data = pandas.read_csv("url_light.csv")

# Build features pandas dataframe
print 'Building url features matrix... \n'
features = []
for index, row in urls_data.iterrows():
    features.append(classifier.classifier(row['url']))

# Format the feature set and combine with url information
features = pandas.DataFrame(features)
urls_data = pandas.concat([urls_data, features], axis=1)

# Labels
y = urls_data["label"]

# Features (minus the url)
x = urls_data.drop(['url','label'], axis=1)

# Split the data for training and testing
x_trn, x_tst, y_trn, y_tst = train_test_split(x, y, test_size=0.5, random_state=42)


def linearregression(x_predict_url):
	if (x_predict_url == None):
		print '\n### Running Linear Regression Algorithm\n'

		# Create Linear Regression classifier object model
		model = linear_model.LinearRegression()

		# Train the model using the training sets and check score
		model.fit(x_trn, y_trn)
		print '\tTraining score\t\t', model.score(x_trn, y_trn)

		# Calculate algorithm scores
		predicted= model.predict(x_tst)
		probabilities_threshold = (predicted > .5).astype(numpy.int)
		avg_prec = average_precision_score(numpy.array(y_tst), probabilities_threshold)
		recall = recall_score(y_tst, probabilities_threshold, average='micro')

		# Print the algorithm scores
		print '\tTesting score\t\t',  model.score(x_tst, predicted)
		print '\tAccuracy score\t\t',  accuracy_score(y_tst, probabilities_threshold)
		print '\tPrecision score:\t' + str(avg_prec)
		print '\tRecall score\t\t' + str(recall)
		
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
		model.fit(x_trn, y_trn)
		print '\tTraining score\t\t', model.score(x_trn, y_trn)

		# Calculate algorithm scores
		predicted= model.predict(x_tst)
		avg_prec = average_precision_score(numpy.array(y_tst), predicted)
		recall = recall_score(y_tst, predicted, average='micro')

			# Print the algorithm scores
		print '\tTesting score\t\t', model.score(x_tst, predicted)
		print '\tAccuracy score\t\t', accuracy_score(y_tst, predicted)
		print '\tPrecision score:\t' + str(avg_prec)
		print '\tRecall score\t\t' + str(recall)
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
		model.fit(x_trn, y_trn)
		print '\tTraining score\t\t', model.score(x_trn, y_trn)

		# Calculate algorithm scores
		predicted= model.predict(x_tst)
		probabilities = model.predict_proba(x_tst)
		probabilities_array = numpy.array((probabilities[:,1]))
		avg_prec = average_precision_score(numpy.array(y_tst), probabilities_array)
		recall = recall_score(y_tst, probabilities_array.round(), average='micro')

			# Print the algorithm scores
		print '\tTesting score\t\t', model.score(x_tst, predicted)
		print '\tAccuracy score\t\t', accuracy_score(y_tst, probabilities_array.round())
		print '\tPrecision score:\t' + str(avg_prec)
		print '\tRecall score\t\t' + str(recall)
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
		model.fit(x_trn, y_trn)
		print '\tTraining score\t\t', model.score(x_trn, y_trn)

		# Calculate algorithm scores
		predicted= model.predict(x_tst)
		probabilities = model.predict_proba(x_tst)
		probabilities_array = numpy.array((probabilities[:,1]))
		probabilities_threshold = (probabilities_array > .5).astype(numpy.int)
		avg_prec = average_precision_score(numpy.array(y_tst), probabilities_threshold)
		recall = recall_score(y_tst, probabilities_threshold, average='micro')

		# Print the algorithm scores
		print '\tTesting score\t\t', model.score(x_tst, probabilities_threshold)
		print '\tAccuracy score\t\t', accuracy_score(y_tst, probabilities_threshold)
		print '\tPrecision score:\t' + str(avg_prec)
		print '\tRecall score\t\t' + str(recall)
		
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

		# Train the model using the training sets and check score
		model.fit(x_trn, y_trn)
		print '\tTraining score\t\t', model.score(x_trn, y_trn)

		# Calculate algorithm scores
		predicted= model.predict(x_tst)
		avg_prec = average_precision_score(numpy.array(y_tst), predicted)
		recall = recall_score(y_tst, predicted, average='micro')

		# Print the algorithm scores
		print '\tTesting score\t\t', model.score(x_tst, predicted)
		print '\tAccuracy score\t\t', accuracy_score(y_tst, predicted)
		print '\tPrecision score:\t' + str(avg_prec)
		print '\tRecall score\t\t' + str(recall)

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
		model.fit(x_trn, y_trn)
		print '\tTraining score\t\t', model.score(x_trn, y_trn)

		# Calculate algorithm scores
		predicted= model.predict(x_tst)
		probabilities = model.predict_proba(x_tst)
		probabilities_array = numpy.array((probabilities[:,1]))
		avg_prec = average_precision_score(numpy.array(y_tst), probabilities_array)
		recall = recall_score(y_tst, predicted, average='micro')

		# Print the algorithm scores
		print '\tTesting score\t\t', model.score(x_tst, y_tst)
		print '\tAccuracy score\t\t', accuracy_score(y_tst, predicted)
		print '\tPrecision score:\t' + str(avg_prec)
		print '\tRecall score\t\t' + str(recall)
		
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
		model = KMeans(n_clusters=8, random_state=0)

		# Train the model using the training sets and check score
		model.fit(x_trn)
		print '\tTraining score\t\t', model.score(x_trn, y_trn)

		# Calculate algorithm scores
		predicted= model.predict(x_tst)
		avg_prec = average_precision_score(numpy.array(y_tst), predicted)
		recall = recall_score(y_tst, predicted, average='micro')

		# Print the algorithm scores
		print '\tTesting score\t\t', model.score(x_tst, predicted)
		print '\tAccuracy score\t\t', accuracy_score(y_tst, predicted)
		print '\tPrecision score:\t' + str(avg_prec)
		print '\tRecall score\t\t' + str(recall)

	
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
		model.fit(x_trn, y_trn)
		print '\tTraining score\t\t', model.score(x_trn, y_trn)

		# Calculate algorithm scores
		predicted= model.predict(x_tst)
		avg_prec = average_precision_score(numpy.array(y_tst), predicted)
		recall = recall_score(y_tst, predicted, average='micro')

		# Print the algorithm scores
		print '\tTesting score\t\t', model.score(x_tst, y_tst)
		print '\tAccuracy score\t\t', accuracy_score(y_tst, predicted)
		print '\tPrecision score:\t' + str(avg_prec)
		print '\tRecall score\t\t' + str(recall)
	
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
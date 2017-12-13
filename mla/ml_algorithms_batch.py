import os

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

# Machine Learning Data Retention
from sklearn.externals import joblib

vectorizer = TfidfVectorizer()

class MachineLearningAlgorithms:

	def __init__(self):
		print '\n### Machine Learning Active\n'

		# Run the machine learning algorithms to prepare models
		for cnt in range(0,2):
			print '\n\n### Machine Learning Round: ' + str(cnt)
			
			# Load Url Data
			urls_data = pandas.read_csv('mla/data_mini/url_' + str(cnt) + '.txt')
			
			# Labels
			y = urls_data["label"]
			
			# Build feature set
			X = urls_data["url"]
			vectorizer = TfidfVectorizer()
			x = vectorizer.fit_transform(X)
			
			# Split the data for training and testing
			self.x_trn, self.x_tst, self.y_trn, self.y_tst = train_test_split(x, y, test_size=0.2, random_state=42)
			self.linearregression()
			self.logisticregression()
			self.decisiontree()
			self.supportvectormachine()
			self.naivebayes()
			self.knearestneighbors()
			self.kmeans()
			self.randomforest()

		print '\nMachine Learning Algorithms Preparation Complete\n\n'

	def linearregression(self):	# Model 1
		print '\n### Running Linear Regression Algorithm\n'

		# Create Linear Regression classifier object model
		modelFile = '/tmp/model_1.joblib.pkl'
		if (os.path.isfile(modelFile)):
			model = self.loadmodel(1)
		else:
			model = linear_model.LinearRegression()

		# Train the model using the training sets and check score
		model.fit(self.x_trn, self.y_trn)
		print '\tTraining score\t\t', model.score(self.x_trn, self.y_trn)

		# Calculate algorithm scores
		predicted = model.predict(self.x_tst)
		probabilities_threshold = (predicted > .5).astype(numpy.int)
		avg_prec = average_precision_score(numpy.array(self.y_tst), probabilities_threshold)
		recall = recall_score(self.y_tst, probabilities_threshold, average='micro')

		# Print the algorithm scores
		print '\tTesting score\t\t',  model.score(self.x_tst, predicted)
		print '\tAccuracy score\t\t',  accuracy_score(self.y_tst, probabilities_threshold)
		print '\tPrecision score:\t' + str(avg_prec)
		print '\tRecall score\t\t' + str(recall)

		self.savemodel(model, 1) # Save the training model

	def logisticregression(self):	# Model 2
		print '\n### Running Logistic Regression Algorithm\n'

		# Create Logistic Regression classifier object model
		modelFile = '/tmp/model_2.joblib.pkl'
		if (os.path.isfile(modelFile)):
			model = self.loadmodel(2)
		else:
			model = LogisticRegression()

		# Train the model using the training sets and check score
		model.fit(self.x_trn, self.y_trn)
		print '\tTraining score\t\t', model.score(self.x_trn, self.y_trn)

		# Calculate algorithm scores
		predicted= model.predict(self.x_tst)
		avg_prec = average_precision_score(numpy.array(self.y_tst), predicted)
		recall = recall_score(self.y_tst, predicted, average='micro')

			# Print the algorithm scores
		print '\tTesting score\t\t', model.score(self.x_tst, predicted)
		print '\tAccuracy score\t\t', accuracy_score(self.y_tst, predicted)
		print '\tPrecision score:\t' + str(avg_prec)
		print '\tRecall score\t\t' + str(recall)

		self.savemodel(model, 2) # Save the training model

	def decisiontree(self):	# Model 3
		print '\n### Running Decision Tree Algorithm\n'

		# Create Decision Tree classifier object model
		modelFile = '/tmp/model_3.joblib.pkl'
		if (os.path.isfile(modelFile)):
			model = self.loadmodel(3)
		else:
			model = tree.DecisionTreeClassifier(criterion='gini') # for classification, here you can change the algorithm as gini or entropy (information gain) by default it is gini  
			# model = tree.DecisionTreeRegressor() for regression

		# Train the model using the training sets and check score
		model.fit(self.x_trn, self.y_trn)
		print '\tTraining score\t\t', model.score(self.x_trn, self.y_trn)

		# Calculate algorithm scores
		predicted= model.predict(self.x_tst)
		probabilities = model.predict_proba(self.x_tst)
		probabilities_array = numpy.array((probabilities[:,1]))
		avg_prec = average_precision_score(numpy.array(self.y_tst), probabilities_array)
		recall = recall_score(self.y_tst, probabilities_array.round(), average='micro')

		# Print the algorithm scores
		print '\tTesting score\t\t', model.score(self.x_tst, predicted)
		print '\tAccuracy score\t\t', accuracy_score(self.y_tst, probabilities_array.round())
		print '\tPrecision score:\t' + str(avg_prec)
		print '\tRecall score\t\t' + str(recall)

		self.savemodel(model, 3) # Save the training model

	def supportvectormachine(self):	# Model 4
		print '\n### Running Support Vector Machine Algorithm\n'

		# Create Support Vector Machine classifier object model
		modelFile = '/tmp/model_4.joblib.pkl'
		if (os.path.isfile(modelFile)):
			model = self.loadmodel(4)
		else:
			model = svm.SVC(probability=True) # there is various option associated with it, this is simple for classification.

		# Train the model using the training sets and check score
		model.fit(self.x_trn, self.y_trn)
		print '\tTraining score\t\t', model.score(self.x_trn, self.y_trn)

		# Calculate algorithm scores
		predicted= model.predict(self.x_tst)
		probabilities = model.predict_proba(self.x_tst)
		probabilities_array = numpy.array((probabilities[:,1]))
		probabilities_threshold = (probabilities_array > .5).astype(numpy.int)
		avg_prec = average_precision_score(numpy.array(self.y_tst), probabilities_threshold)
		recall = recall_score(self.y_tst, probabilities_threshold, average='micro')

		# Print the algorithm scores
		print '\tTesting score\t\t', model.score(self.x_tst, probabilities_threshold)
		print '\tAccuracy score\t\t', accuracy_score(self.y_tst, probabilities_threshold)
		print '\tPrecision score:\t' + str(avg_prec)
		print '\tRecall score\t\t' + str(recall)

		self.savemodel(model, 4) # Save the training model

	def naivebayes(self):	# Model 5
		print '\n### Running Naive Bayes Algorithm\n'

		# Create Naive Bayes classifier object model
		modelFile = '/tmp/model_5.joblib.pkl'
		if (os.path.isfile(modelFile)):
			model = self.loadmodel(5)
		else:
			model = GaussianNB()

		# Train the model using the training sets and check score
		model.fit(self.x_trn.toarray(), self.y_trn)
		print '\tTraining score\t\t', model.score(self.x_trn.toarray(), self.y_trn)

		# Calculate algorithm scores
		predicted= model.predict(self.x_tst.toarray())
		avg_prec = average_precision_score(numpy.array(self.y_tst), predicted)
		recall = recall_score(self.y_tst, predicted, average='micro')

		# Print the algorithm scores
		print '\tTesting score\t\t', model.score(self.x_tst.toarray(), predicted)
		print '\tAccuracy score\t\t', accuracy_score(self.y_tst, predicted)
		print '\tPrecision score:\t' + str(avg_prec)
		print '\tRecall score\t\t' + str(recall)

		self.savemodel(model, 5) # Save the training model

	def knearestneighbors(self):	# Model 6
		print '\n### Running K-Nearest Neighbors Algorithm\n'

		# Create K-Nearest Neighbors classifier object model
		modelFile = '/tmp/model_6.joblib.pkl'
		if (os.path.isfile(modelFile)):
			model = self.loadmodel(6)
		else:
			model = KNeighborsClassifier(n_neighbors=5, leaf_size=25) # default value for n_neighbors is 5

		# Train the model using the training sets and check score
		model.fit(self.x_trn, self.y_trn)
		print '\tTraining score\t\t', model.score(self.x_trn, self.y_trn)

		# Calculate algorithm scores
		predicted= model.predict(self.x_tst)
		probabilities = model.predict_proba(self.x_tst)
		probabilities_array = numpy.array((probabilities[:,1]))
		avg_prec = average_precision_score(numpy.array(self.y_tst), probabilities_array)
		recall = recall_score(self.y_tst, predicted, average='micro')

		# Print the algorithm scores
		print '\tTesting score\t\t', model.score(self.x_tst, self.y_tst)
		print '\tAccuracy score\t\t', accuracy_score(self.y_tst, predicted)
		print '\tPrecision score:\t' + str(avg_prec)
		print '\tRecall score\t\t' + str(recall)

		self.savemodel(model, 6) # Save the training model

	def kmeans(self):	# Model 7
		print '\n### Running K-Means Algorithm\n'

		# Create K-Means classifier object model
		modelFile = '/tmp/model_7.joblib.pkl'
		if (os.path.isfile(modelFile)):
			model = self.loadmodel(7)
		else:
			model = KMeans(n_clusters=8, random_state=0)

		# Train the model using the training sets and check score
		model.fit(self.x_trn)
		print '\tTraining score\t\t', model.score(self.x_trn, self.y_trn)

		# Calculate algorithm scores
		predicted= model.predict(self.x_tst)
		avg_prec = average_precision_score(numpy.array(self.y_tst), predicted)
		recall = recall_score(self.y_tst, predicted, average='micro')

		# Print the algorithm scores
		print '\tTesting score\t\t', model.score(self.x_tst, predicted)
		print '\tAccuracy score\t\t', accuracy_score(self.y_tst, predicted)
		print '\tPrecision score:\t' + str(avg_prec)
		print '\tRecall score\t\t' + str(recall)

		self.savemodel(model, 7) # Save the training model

	def randomforest(self):	# Model 8
		print '\n### Running Random Forest Algorithm\n'

		# Create Random Forest classifier object model
		modelFile = '/tmp/model_8.joblib.pkl'
		if (os.path.isfile(modelFile)):
			model = self.loadmodel(8)
		else:
			model = RandomForestClassifier()

		# Train the model using the training sets and check score
		model.fit(self.x_trn, self.y_trn)
		print '\tTraining score\t\t', model.score(self.x_trn, self.y_trn)

		# Calculate algorithm scores
		predicted= model.predict(self.x_tst)
		avg_prec = average_precision_score(numpy.array(self.y_tst), predicted)
		recall = recall_score(self.y_tst, predicted, average='micro')

		# Print the algorithm scores
		print '\tTesting score\t\t', model.score(self.x_tst, self.y_tst)
		print '\tAccuracy score\t\t', accuracy_score(self.y_tst, predicted)
		print '\tPrecision score:\t' + str(avg_prec)
		print '\tRecall score\t\t' + str(recall)

		self.savemodel(model, 8) # Save the training model

	def savemodel(self, model, algorithm):
		modelFile = '/tmp/model_' + str(algorithm) + '.joblib.pkl'
		joblib.dump(model, modelFile, compress=9)

	def loadmodel(self, algorithm):
		modelFile = '/tmp/model_' + str(algorithm) + '.joblib.pkl'
		return joblib.load(modelFile)

	def queryurl(self, var_url):

		# Build feature set
		X = [var_url]
		x = vectorizer.transform(X)

		# Run the prediction algorithms
		return self.queryalgorithm(x)

	def queryalgorithm(self, var_set):
		predictions = [] # Track algorithm predictions

		for x in range(1,9):
			model = self.loadmodel(str(x))
			if (x == 1):
				# Model 1 - Linear Regression
				new_predict = model.predict(var_set)
				new_predict = (new_predict > .6).astype(numpy.int)
			elif (x == 4):
				# Model 4 - Support Vector Machine
				new_predict = model.predict(var_set)
				new_predict_prob = model.predict_proba(var_set)
				new_predict_prob_arr = numpy.array((new_predict_prob[:,1]))
				new_predict = (new_predict_prob_arr > .5).astype(numpy.int)
			elif (x == 5):
				# Model 5 - Naive Bayes
				new_predict = model.predict(var_set.toarray())
			else:
				# Model 2 - Logistic Regression
				# Model 3 - Decision Tree
				# Model 6 - K-Nearest Neighbor
				# Model 7 - K-Means
				# Model 8 - Random Forest
				new_predict = model.predict(var_set)

			if (x != 7):	# Ignore K-Means (returns cluster number)
				predictions.append(new_predict)

		# Block url if 4 of 7 algorithms indicate it may be malicious
		if (sum(predictions) > 3):
			return True

		return False

	def removemodels(self):
		for x in range(1,9):
			model = '/tmp/model_' + str(x) + '.joblib.pkl'
			os.remove(model)

if __name__ == "__main__":
	# Create a new MLA instance
    mla = MachineLearningAlgorithms()

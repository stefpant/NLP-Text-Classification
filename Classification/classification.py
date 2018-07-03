from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.naive_bayes import MultinomialNB
from sklearn import preprocessing, svm
from sklearn.preprocessing import MinMaxScaler
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk import FreqDist
from my_knn_classifier import knn_find_stats

import numpy as np
import pandas as pd

import sys
reload(sys)
sys.setdefaultencoding('utf8')

import warnings
warnings.filterwarnings("ignore")

def simple_lsi_transform(Array, w, num_comp, stopWords):
	sw = ENGLISH_STOP_WORDS.union(stopWords)
	count_vectorizer = CountVectorizer(stop_words=stopWords)
	tfidf_transformer = TfidfTransformer()
	lsi_transformer = TruncatedSVD(n_components=num_comp)
	CVArray = count_vectorizer.fit_transform(Array)
	TFIDFArray = tfidf_transformer.fit_transform(CVArray).multiply(w)
	ArrayLSI = lsi_transformer.fit_transform(TFIDFArray)
	#ArrayLSI = lsi_transformer.fit_transform(CVArray)
	return ArrayLSI

def preprocess_data(Array):
	tokenizer = RegexpTokenizer(r'\w+')
	lemmatizer = WordNetLemmatizer()
	retArray = []
	allwords = []
	for i in range(len(Array)):
		tokens = tokenizer.tokenize(Array[i])
		lemtokens = [lemmatizer.lemmatize(f.lower(),pos="v") for f in tokens]
#		for j in lemtokens:
#			if j not in ENGLISH_STOP_WORDS :
#				allwords.append(j)
		retArray.append(" ".join(lemtokens))
#	allwordDist = FreqDist(allwords)
#	nlen = int(len(allwordDist)/20)
#	mostCommon = allwordDist.most_common(nlen)
#	mcList = []
#	for i in mostCommon:
#		if i[1] > 1*len(Array):
#			mcList.append(i[0])
#		else :
#			break
	return retArray


#returns as tuple the evaluations of clf classifier
def find_stats(X ,y ,clf, cvm):
	pred_y = cross_val_predict(clf,X,y,cv=cvm)
	acc = accuracy_score(y,pred_y)
	rec = recall_score(y,pred_y,average='macro')
	prec = precision_score(y,pred_y,average='macro')
	f1 = f1_score(y,pred_y,average='macro')
	return (acc,prec,rec,f1)

def gridSearchSVM(X, y):
	parameters = {'kernel':('linear','rbf'),'C':[0.1, 1, 10],'gamma':[0.001, 0.01, 0.1, 1]}
	svc = svm.SVC(probability=True)
	clf = GridSearchCV(svc, parameters, cv=10)
	Xscale = preprocessing.scale(X)
	clf.fit(Xscale,y)
	best = clf.best_params_
	print "kernel:",best['kernel']
	print "C:",best['C']
	print "gamma:",best['gamma']
	return

def my_stop_words():
	sw = ['s','say','make','t','time','year','years','people','new','party','come',
		'think','just','want','know','use','world','look','need','way','leave','tell',
		'end','don','change','good','1','2','10','0','000','set','home','house','really','m',
		'day','add','ve','best','live','support','include','4','5','break','enter','22',
		'plus','mother','greece','100','single','total''mean','open','number','city'
		'share','meet','ask','try','high','little','plan','lot','cut','face','big',
		'uk','far','better','help','week','work','point','expect','win','right']
	return sw


if __name__ =="__main__":

	traindf = pd.read_csv('../datasets/train_set.csv', sep="\t", encoding='utf-8')
	testdf = pd.read_csv('../datasets/test_set.csv', sep="\t", encoding='utf-8')

	le=preprocessing.LabelEncoder()
	le.fit(traindf["Category"])
	y = le.transform(traindf["Category"])


	trainContent = [i for i in traindf.Content]
	#trainTitle = [i for i in traindf.Title]
	testContent = [i for i in testdf.Content]
	#testTitle = [i for i in testdf.Title]


	NComps = 50#here testing mun of components
	XContentLSI = simple_lsi_transform(trainContent,1,NComps,[])
	X = XContentLSI

	#cross validation method = Stratified-10-Fold
	#used in cross_val_predict to train-test our train datas
	folds = 10
	cvm = StratifiedKFold(n_splits=folds,shuffle=False)

	#
	#gridSearchSVM(X, y)
	#
	#best params from grid search:
	#	kernel: rbf
	#	C:10
	#	gamma:1

	stats = []
	classifiers = []
	print 'mnb'
	#multinomial naive-bayes
	clf = MultinomialNB()
	rangeMax = 20
	scaler = MinMaxScaler(feature_range=(-50,50))
	Xmnb = scaler.fit_transform(X)#transform lsi to remove negative values
	stats.append(find_stats(Xmnb, y, clf, cvm))
	classifiers.append('Naive Bayes')
	print 'rf'
	#random forests
	clf = RandomForestClassifier(n_estimators=30)
	stats.append(find_stats(X, y, clf, cvm))
	classifiers.append('Random Forest')
	print 'svm'
	#support vector machines
	clf = svm.SVC(kernel='rbf',C=10,gamma=1,probability=True)
	stats.append(find_stats(X, y, clf, cvm))
	classifiers.append('SVM')

	print 'knn'
	#k-nearest neighbor
	k = 10
	stats.append(knn_find_stats(X, y, k, cvm))
	classifiers.append('KNN')
	print 'mm'
	#my method
	#data preprocessing
	trainContent = preprocess_data(trainContent)
	#prepXT = preprocess_data(trainTitle)
	XContentLSI = simple_lsi_transform(trainContent,1,NComps,my_stop_words())
	X = XContentLSI
	Xmnb = scaler.fit_transform(X)#transform lsi to remove negative values
	#XTitleLSI = simple_lsi_transform(trainTitle,1,NComps,my_stop_words())
	#X = np.hstack((XContentLSI,XTitleLSI))#concat np.arrays

	clf = svm.SVC(kernel='rbf',C=10,gamma=1,probability=True)
	stats.append(find_stats(X, y, clf, cvm))
	classifiers.append('My Method')

	dict1 ={'Statistic Measure':['Accuracy','Precision','Recall','F-Measure'],
			'Naive Bayes':stats[0],
			'Random Forest':stats[1],
			'SVM':stats[2],
			'KNN':stats[3],
			'My Method':stats[4]
			}

	df = pd.DataFrame.from_dict(data=dict1)
	#reordering
	df = df[['Statistic Measure','Naive Bayes','Random Forest','SVM','KNN','My Method']]
	df.to_csv('./csv_results/EvaluationMetric_10Fold.csv',sep='\t',index=False)

	#test data preprocessing
	testContent = preprocess_data(testContent)
	#prepXT = preprocess_data(testTitle)
	ZContentLSI = simple_lsi_transform(testContent,1,NComps,my_stop_words())
	Z = ZContentLSI
	Zmnb = scaler.fit_transform(Z)#transform lsi to remove negative values
	#ZTitleLSI = simple_lsi_transform(prepXT,1,NComps,my_stop_words())
	#Z = np.hstack((ZContentLSI,ZTitleLSI))

#in comments csv format for kaggle's testing
	#clf from my method
	clf.fit(Xmnb,y)#train with X, predict Z....lets see results
	y_pred = clf.predict(Zmnb)
	test_categories = le.inverse_transform(y_pred)
	test_ids = [i for i in testdf.Id]

#	dict2 = {'ID':test_ids,
#			 'Predicted_Category':test_categories}
	dict2 = {'Id':test_ids,
			 'Category':test_categories}

	df = pd.DataFrame.from_dict(data=dict2)
#	df = df[['ID','Predicted_Category']]#reordering
#	df.to_csv('testSet_categories.csv',sep='\t',index=False)
	df = df[['Id','Category']]#reordering
	df.to_csv('./csv_results/testSet_categories.csv',index=False)#by default sep=','


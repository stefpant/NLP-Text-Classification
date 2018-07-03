from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics.pairwise import euclidean_distances
from operator import itemgetter
from scipy.spatial import distance
import numpy as np
import pandas as pd

def knn_predict(X_train,X_test,y_train,k):
	predictions = []
	for i in range(X_test.shape[0]):
		dists = []
		for j in range(X_train.shape[0]):#find and save for test[i] all dists from train
			jdist = distance.euclidean(X_test[i],X_train[j])
			dists.append((jdist,y_train[j]))
		dists.sort(key=itemgetter(0))
		if len(dists) > k :
			dists = dists[:k]#keep first k neighbors
		categories = [x[1] for x in dists]
		#for x in dists:
		#	categories.append(x[1])
		mycategory = max(set(categories),key=categories.count)#majority voting
		predictions.append(mycategory)
	return np.array(predictions)


#actualy here do something like cros_val_predict
#and using the above function as knn's predictions
def knn_find_stats( X, y, k, cvm):
	k_acc = []
	k_prec = []
	k_rec = []
	k_f1 = []
	#cvm.split splits the dataset X,y in k(10) subsets(actualy only their indices)
	#and returns those indices in 2 list[k] to let us split the arrays
	for train_indices,test_indices in cvm.split(X,y):
		#print 'done'
		X_train,y_train = X[train_indices],y[train_indices]
		X_test,y_test = X[test_indices],y[test_indices]
		pred_y = knn_predict(X_train,X_test,y_train,k)
		k_acc.append(accuracy_score(y_test,pred_y))
		k_prec.append(precision_score(y_test,pred_y,average='macro'))
		k_rec.append(recall_score(y_test,pred_y,average='macro'))
		k_f1.append(f1_score(y_test,pred_y,average='macro'))

	acc = np.mean(k_acc)
	prec = np.mean(k_prec)
	rec = np.mean(k_rec) 
	f1 = np.mean(k_f1)

	return (acc,prec,rec,f1)

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn import preprocessing
from wordcloud import WordCloud , STOPWORDS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train_data = pd.read_csv('../datasets/train_set.csv', sep="\t")

train_data = train_data[0:25]

for i in set(train_data["Category"]):
	mydata = []
	for j in range(len(train_data)):
		if train_data["Category"][j]==i :
			mydata.append([train_data["Content"][j]])
	joindata = ""
	realjoin = ""
	for x in mydata:
		joindata = ' '.join(x)
		realjoin = realjoin + " " + joindata

	if len(mydata) != 0 :
		wordcloud = WordCloud(stopwords=ENGLISH_STOP_WORDS,max_font_size=40, relative_scaling=.5).generate(realjoin)

		fig = plt.figure()
		plt.imshow(wordcloud)
		plt.axis("off")
	
		fig.savefig("./images/"+ i +".png")
	plt.close

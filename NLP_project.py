#!/usr/bin/env python
# coding: utf-8

# In[12]:


import zipfile
import os
import re


# In[72]:


os.chdir(r"C:\Users\errav\Downloads")


# In[12]:


zip_path = r"C:\Users\errav\Downloads\txt_reviews.zip"

# Specify the directory where you want to extract the contents
extracted_dir = r"C:\Users\errav\Downloads"

# Unzip the contents of the zip file folder
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extracted_dir) 


# In[10]:


import os
import pandas as pd

text_files_directory = r"C:\Users\errav\Downloads\txt_reviews"

file_names = []
file_contents = []

for filename in os.listdir(text_files_directory):
    if filename.endswith(".txt"):
        file_path = os.path.join(text_files_directory, filename)

        with open(file_path, 'r', encoding='utf-8') as file:
            file_name = os.path.splitext(filename)[0] 
            content = file.read()
            
            file_names.append(file_name)
            file_contents.append(content)

# Create a DataFrame
df = pd.DataFrame({'File Name': file_names, 'File Content': file_contents})

# Display the DataFrame
df.head()


# In[ ]:





# In[11]:


len(df['File Name'])


# In[13]:


df['File Content'][0]


# In[14]:


regex = r"ProductId: (.*)\nUserId:"
df['ProductId'] = df['File Content'].apply(lambda x: re.findall(regex,x)[0])


# In[15]:


regex = r"UserId: (.*)\nProfileName:"
df['Userid'] = df['File Content'].apply(lambda x: re.findall(regex,x)[0])


# In[16]:


regex = r"ProfileName: (.*)\nHelpfulnessNumerator:"
df['ProfileName'] = df['File Content'].apply(lambda x: re.findall(regex,x)[0])


# In[17]:


regex = r"HelpfulnessNumerator: (.*)\nHelpfulnessDenominator:"
df['HelpfulnessNumerator'] = df['File Content'].apply(lambda x: re.findall(regex,x)[0])


# In[18]:


regex = r"HelpfulnessDenominator: (.*)\nScore:"
df['HelpfulnessDenominator'] = df['File Content'].apply(lambda x: re.findall(regex,x)[0])


# In[19]:


regex = r"Score: (.*)\nTime:"
df['Score'] = df['File Content'].apply(lambda x: re.findall(regex,x)[0])


# In[20]:


regex = r"Time: (.*)\nReviewSummary:"
df['Time'] = df['File Content'].apply(lambda x: re.findall(regex,x)[0])


# In[21]:


regex = r"ReviewSummary: (.*)\nReviewText:"
df['ReviewSummary'] = df['File Content'].apply(lambda x: re.findall(regex,x)[0])


# In[22]:


regex = r"ReviewText: (.*)\n"
df['ReviewText'] = df['File Content'].apply(lambda x: re.findall(regex,x)[0])


# In[23]:


final_df=pd.DataFrame({"ProductId":df["ProductId"],"UserId":df['Userid'],"ProfileName":df['ProfileName']
                       ,"HelpfulnessNumerator":df['HelpfulnessNumerator'],"HelpfulnessDenominator":df['HelpfulnessDenominator'],
                      "Score":df['Score'],"Time":df['Time'],"ReviewSummary":df['ReviewSummary'],"ReviewText":df['ReviewText']})


# In[24]:


final_df.head()


# In[25]:


final_df.to_csv("TextDataProject.csv")


# In[26]:


final_df=pd.read_csv("TextDataProject.csv")


# In[27]:


final_df=final_df.drop("Unnamed: 0",axis=1)


# In[28]:


final_df.insert(0, 'Id', range(1,len(final_df)+1))


# In[29]:


final_df.dtypes


# In[30]:


final_df['HelpfulnessNumerator']=final_df['HelpfulnessNumerator'].astype('int')


# In[31]:


final_df['HelpfulnessDenominator']=final_df['HelpfulnessDenominator'].astype('int')


# In[32]:


final_df['Score']=final_df['Score'].astype('int')


# In[33]:


final_df.dtypes


# In[34]:


final_df[final_df.duplicated()]


# In[35]:


final_df.insert(6, 'Helpfulness', final_df['HelpfulnessNumerator']/final_df['HelpfulnessDenominator'])


# In[36]:


final_df['Helpfulness'].value_counts()


# In[37]:


final_df['Helpfulness'].isnull().sum()


# In[38]:


final_df['Helpfulness'].median()


# In[39]:


final_df['Helpfulness']=final_df['Helpfulness'].fillna(final_df['Helpfulness'].median())


# In[40]:


final_df['ProfileName']=final_df['ProfileName'].fillna(final_df['ProfileName'].mode()[0])
final_df['ReviewSummary']=final_df['ReviewSummary'].fillna(final_df['ReviewSummary'].mode()[0])


# In[41]:


final_df.isnull().sum()


# In[42]:


final_df['Combine_Review']= final_df['ReviewSummary'] +' ' + final_df['ReviewText']


# In[43]:


final_df.to_csv("NLP_Project.csv")


# In[44]:


final_df=pd.read_csv("NLP_Project.csv")


# In[45]:


final_df=final_df.drop("Unnamed: 0",axis=1)


# In[70]:


final_df


# In[69]:


from nltk.tokenize import sent_tokenize, word_tokenize
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob , Word
import matplotlib.pyplot as plt
import seaborn as sns


# In[68]:


from nltk import *


# In[56]:


plt.figure(figsize=(8,5))
sns.countplot(x="Score",data=final_df)
plt.title("Number of stars by Score")
plt.show()


# In[66]:


from tqdm import tqdm, tqdm_notebook
#!pip install nltk


# In[67]:


tqdm.pandas()


# In[70]:


lemmatize=WordNetLemmatizer()


# In[71]:


def clean_text(doc, stem=False):
    # Removing Special Characters
    doc = re.sub(r'[^A-z0-9 ]','',doc)
    doc = re.sub(r'_','',doc)
    # Convert to Lowercase
    doc = doc.lower()
    # Split the document into tokens
    doc_list = doc.split()
    # Correct the Spellings and removing stop words
    doc_list = [word for word in doc_list if word not in stopwords.words('english')]
    #doc_list = [str(TextBlob(word).correct()) for word in doc_list if word not in sw]
    # Convert to Singularize
    #doc_list = [str(Word(word).singularize()) for word in doc_list]
    
    return " ".join(doc_list)


# In[72]:


import nltk
#nltk.download('stopwords')


# In[73]:


final_df['Text_cleaned_lemma'] = final_df['Combine_Review'].progress_apply(clean_text)


# In[74]:


final_df


# In[80]:


final_df.to_csv("NLP_text_Project.csv")


# In[81]:


import pandas as pd
final_df=pd.read_csv("NLP_Project.csv")


# In[82]:


final_df


# In[83]:


final_df=final_df.drop("Unnamed: 0",axis=1)


# In[84]:


final_df


# In[9]:


get_ipython().system(' pip install wordcloud')


# In[10]:


from wordcloud import WordCloud
from wordcloud import STOPWORDS


# In[11]:


# Create a new data frame "reviews" to perform exploratory data analysis upon that
reviews = final_df
# Dropping null values
reviews.dropna(inplace=True)


# In[12]:


score_1 = reviews[reviews['Score'] == 1]
score_2 = reviews[reviews['Score'] == 2]
score_3 = reviews[reviews['Score'] == 3]
score_4 = reviews[reviews['Score'] == 4]
score_5 = reviews[reviews['Score'] == 5]


# In[13]:


reviews_sample = pd.concat([score_1,score_2,score_3,score_4,score_5],axis=0)
reviews_sample.reset_index(drop=True,inplace=True)


# In[14]:


reviews_sample


# In[15]:


word_string=" ".join(reviews_sample['ReviewText'].str.lower())


# In[16]:


import matplotlib.pyplot as plt
wordcloud = WordCloud(background_color='white').generate(word_string)
plt.figure(figsize=(10,10))
plt.imshow(wordcloud,interpolation='bilinear')
plt.axis("off")
plt.show()


# In[17]:


# Now let's split the data into Negative (Score is 1 or 2) and Positive (4 or #5) Reviews.
negative_reviews = reviews_sample[reviews_sample['Score'].isin([1,2]) ]
positive_reviews = reviews_sample[reviews_sample['Score'].isin([4,5]) ]
# Transform to single string
negative_reviews_str = " ".join(negative_reviews['ReviewText'].str.lower())
positive_reviews_str = " ".join(positive_reviews['ReviewText'].str.lower())


# In[18]:


wordcloud_negative = WordCloud(background_color='white').generate(negative_reviews_str)
wordcloud_positive = WordCloud(background_color='black').generate(positive_reviews_str)
# Plot
fig = plt.figure(figsize=(10,10))
ax1 = fig.add_subplot(211)
ax1.imshow(wordcloud_negative,interpolation='bilinear')
ax1.axis("off")
ax1.set_title('Reviews with Negative Scores',fontsize=20)


# In[19]:


fig = plt.figure(figsize=(10,10))
ax2 = fig.add_subplot(212)
ax2.imshow(wordcloud_positive,interpolation='bilinear')
ax2.axis("off")
ax2.set_title('Reviews with Positive Scores',fontsize=20)
plt.show()


# Sentiment Analysis: Pretrained model takes the input from the text description and outputs the sentiment score ranging from -1 to +1 for each sentence
# 
# 

# VADER (Valence Aware Dictionary and sEntiment Reasoner) is a lexicon and rule-based sentiment analysis tool that is specifically attuned to sentiments expressed in social media. VADER uses a combination of A sentiment lexicon is a list of lexical features (e.g., words) which are generally labeled according to their semantic orientation as either positive or negative. VADER not only tells about the Positive and Negative score but also tells us about how positive or negative a sentiment is.
# 
# 

# In[20]:


get_ipython().system('pip install vaderSentiment')


# In[22]:


import re
import os
import sys
import ast
import seaborn as sns
plt.style.use('fivethirtyeight')
# Function for getting the sentiment
cp = sns.color_palette()
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()


# In[24]:


emptyline=[]
for row in reviews['ReviewText']:
    
    vs=analyzer.polarity_scores(row)
    emptyline.append(vs)
# Creating new dataframe with sentiments
df_sentiments=pd.DataFrame(emptyline)
df_sentiments.head()


# In[26]:


# Merging the sentiments back to reviews dataframe
df_c = pd.concat([reviews.reset_index(drop=True), df_sentiments], axis=1)
df_c.head(3)


# In[28]:


# Convert scores into positive and negetive sentiments using some threshold
import numpy as np
df_c['Sentiment'] = np.where(df_c['compound'] >= 0 , 'Positive','Negative')
df_c.head(5)


# In[29]:


result=df_c['Sentiment'].value_counts()
result.plot(kind='bar', rot=0, color=['plum','cyan']);


# In[85]:


final_df.to_csv("NLP_textProject.csv")


# In[86]:


final_df=pd.read_csv("NLP_text_Project.csv")
final_df=final_df.drop("Unnamed: 0",axis=1)
final_df


# In[87]:


from sklearn.model_selection import train_test_split


# In[135]:


X_train,X_test, Y_train,Y_test = train_test_split(final_df['Text_cleaned_lemma'], final_df['Score'],\
                                                 test_size=0.3, random_state=100)


# In[89]:


from sklearn.feature_extraction.text import TfidfVectorizer
tv=TfidfVectorizer()


# In[90]:


from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()


# In[91]:


X_train_processed_bow = cv.fit_transform(X_train)
X_train_processed_tfidf = tv.fit_transform(X_train)


# In[92]:


X_train_processed_bow


# In[93]:


X_train_processed_tfidf


# In[94]:


from sys import getsizeof


# In[95]:


getsizeof(X_train_processed_tfidf)


# In[96]:


X_test_processed_bow = cv.transform(X_test)
X_test_processed_tfidf = tv.transform(X_test)


# # BUILDING THE MODEL

# #  MULTINOMIAL_NB ALGORITHM USING BAG OF WORDS
# 

# In[97]:


from sklearn.naive_bayes import MultinomialNB


# In[98]:


from sklearn.metrics import accuracy_score,confusion_matrix,precision_score, recall_score


# In[99]:


mnb=MultinomialNB()
mnb.fit(X_train_processed_bow,Y_train)


# In[102]:


y_pred_mnb=mnb.predict(X_test_processed_bow)
y_pred_mnb


# In[103]:


accuracy_score(Y_test,y_pred_mnb)


# In[104]:


confusion_matrix(Y_test,y_pred_mnb)


# In[105]:


pd.crosstab(Y_test, y_pred_mnb, rownames=['Actual'], colnames=['Predicted'])


# # DECISION TREE ALGORITHM USING BAG OF WORDS

# In[106]:


get_ipython().run_cell_magic('time', '', 'from sklearn.tree import DecisionTreeClassifier\ndtc=DecisionTreeClassifier()\ndtc.fit(X_train_processed_bow,Y_train)\n')


# In[107]:


get_ipython().run_cell_magic('time', '', 'y_pred_dtc=dtc.predict(X_test_processed_bow)\ny_pred_dtc\n')


# In[108]:


accuracy_score(Y_test,y_pred_dtc)


# In[109]:


confusion_matrix(Y_test,y_pred_dtc)


# In[110]:


pd.crosstab(Y_test, y_pred_dtc, rownames=['Actual'], colnames=['Predicted'])


# # RANDOM FOREST ALGORITHM USING BAG OF WORDS

# In[111]:


get_ipython().run_cell_magic('time', '', 'from sklearn.ensemble import RandomForestClassifier\nrfc=RandomForestClassifier(n_jobs=-1)\nrfc.fit(X_train_processed_bow,Y_train)\n')


# In[115]:


get_ipython().run_cell_magic('time', '', 'y_pred_rfc=rfc.predict(X_test_processed_bow)\ny_pred_rfc\n')


# In[113]:


accuracy_score(Y_test,y_pred_rfc)


# In[114]:


confusion_matrix(Y_test,y_pred_rfc)


# In[116]:


pd.crosstab(Y_test, y_pred_rfc, rownames=['Actual'], colnames=['Predicted'])


# # LOGISTIC REGRESION ALGORITHM USING BAG OF WORDS

# In[117]:


get_ipython().run_cell_magic('time', '', 'from sklearn.linear_model import LogisticRegression\nlrc=LogisticRegression(n_jobs=-1)\nlrc.fit(X_train_processed_bow,Y_train)\n')


# In[118]:


get_ipython().run_cell_magic('time', '', 'y_pred_lrc=lrc.predict(X_test_processed_bow)\ny_pred_lrc\n')


# In[119]:


accuracy_score(Y_test,y_pred_lrc)


# In[120]:


confusion_matrix(Y_test,y_pred_lrc)


# In[121]:


pd.crosstab(Y_test, y_pred_lrc, rownames=['Actual'], colnames=['Predicted'])


# # K-NEAREST NEIGHBORS ALGORITHM USING BAG OF WORDS

# In[122]:


get_ipython().run_cell_magic('time', '', 'from sklearn.neighbors import KNeighborsClassifier\nknc=KNeighborsClassifier()\nknc.fit(X_train_processed_bow,Y_train)\n')


# In[123]:


get_ipython().run_cell_magic('time', '', 'y_pred_knc=knc.predict(X_test_processed_bow)\ny_pred_knc\n')


# In[124]:


accuracy_score(Y_test,y_pred_knc)


# In[125]:


confusion_matrix(Y_test,y_pred_knc)


# In[126]:


pd.crosstab(Y_test, y_pred_knc, rownames=['Actual'], colnames=['Predicted'])


# In[127]:


# Import style 1 (Without Alias)
import pickle

# Import style 2 (With Alias)
import pickle as pk


# In[129]:


# Saving model to pickle file
with open("desired-model-file-name.pkl", "wb") as file: # file is a variable for storing the newly created file, it can be anything.
    pickle.dump(rfc, file) # Dump function is used to write the object into the created file in byte format.


# In[130]:


# Saving model to pickle file
with open("desired-model-file-name.pkl", "wb") as file: # file is a variable for storing the newly created file, it can be anything.
    pickle.dump(knc, file) # Dump function is used to write the object into the created file in byte format.


# In[131]:


# Saving model to pickle file
with open("desired-model-file-name.pkl", "wb") as file: # file is a variable for storing the newly created file, it can be anything.
    pickle.dump(lrc, file) # Dump function is used to write the object into the created file in byte format.


# In[132]:


# Saving model to pickle file
with open("desired-model-file-name.pkl", "wb") as file: # file is a variable for storing the newly created file, it can be anything.
    pickle.dump(dtc, file) # Dump function is used to write the object into the created file in byte format.


# In[133]:


# Saving model to pickle file
with open("desired-model-file-name.pkl", "wb") as file: # file is a variable for storing the newly created file, it can be anything.
    pickle.dump(mnb, file) # Dump function is used to write the object into the created file in byte format.


# In[ ]:





# In[144]:


model=[("Logistic Regression",accuracy_score(Y_test,y_pred_lrc)),("KNN Classification",accuracy_score(Y_test,y_pred_knc)),
       ("Decision Tree Classification",accuracy_score(Y_test,y_pred_dtc)),("Multinomial NB Classification",accuracy_score(Y_test,y_pred_mnb)),
       ("Random Forest Classification",accuracy_score(Y_test,y_pred_rfc))]
    


# In[139]:


predict=pd.DataFrame(data=model,columns=["Algorithm","Accuracy_Score"],index=[1,2,3,4,5])
predict.sort_values(by='Accuracy_Score',ascending=False)


# In[140]:


plt.figure(figsize=(8,3))
sns.barplot(x=predict['Accuracy_Score'],y=predict['Algorithm'])
plt.show()


# In[16]:


import pandas as pd
x = [1 , 2 , 3 , 4 , 5]
d = {'algo' : ['knn' , 'dtc' , 'lr' , 'mnb' , 'rfc'] , 'time' : ['172ms' , '3hrs' , '40m' , '179 ms' , '8hrs']}
df1= pd.DataFrame(d)


# In[17]:


df1


# In[ ]:





# In[ ]:





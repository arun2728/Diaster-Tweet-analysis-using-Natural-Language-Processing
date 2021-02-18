#!/usr/bin/env python
# coding: utf-8

# # Diaster Tweet

# ## Problem Statement:
# 
# Twitter has become an important communication channel in times of emergency.
# The ubiquitousness of smartphones enables people to announce an emergency they’re observing in real-time. Because of this, more agencies are interested in programatically monitoring Twitter (i.e. disaster relief organizations and news agencies).
# 
# But, it’s not always clear whether a person’s words are actually announcing a disaster.
# 
# 

# ## Data Definition:
# 
# **Independent Variable**
# 
# 1. id - a unique identifier for each tweet
# 2. text - the text of the tweet
# 3. location - the location the tweet was sent from (may be blank)
# 4. keyword - a particular keyword from the tweet (may be blank)
# 
# **Dependent Variable**
# 
# - target - in train.csv only, this denotes whether a tweet is about a real disaster (1) or not (0)
# 

# <a id='toc'></a>
# ## **Tabel of Contents**

# 

# ### Import packages

# In[37]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
import tensorflow as tf
import nltk
nltk.download('stopwords')
sns.set()


# ### Load Data

# In[38]:


raw_data = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Disaster Twitter/twitter_data.csv')

raw_data.head()


# In[39]:


print('There are {} rows and {} columns in dataset'.format(raw_data.shape[0],raw_data.shape[1]))


# ### Class distribution

# In[40]:


labels = raw_data.target.value_counts()

sns.barplot(x = labels.index, y = labels)
plt.gca().set_ylabel('Number of samples')
plt.gca().set_xlabel('Target')


# *Note : There are more tweets with class 0 (No disaster) than class 1 (disaster tweets)*

# ### EDA (Exploratory Data Analysis)

# 1. Number of characters in tweets

# In[41]:


fig, (ax1,ax2) = plt.subplots(nrows = 1,ncols = 2,figsize = (10,5))

no_char_0 = list(raw_data[raw_data['target'] == 0]['text'].map(len))
sns.distplot(no_char_0, ax = ax1, color = 'green')


no_char_1 = list(raw_data[raw_data['target'] == 1]['text'].map(len))
sns.distplot(no_char_1, ax = ax2, color = 'red')

ax1.set_title('No disaster tweets')
ax2.set_title('Diaster  tweets')

fig.suptitle('Number of charcters in tweets')
#plt.show()


# **2. Number of words in tweet**

# In[42]:


raw_data.iloc[15]


# In[43]:


fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,5))

no_words_0 = raw_data[raw_data['target'] == 0]['text'].str.split().map(lambda x: len(x))
ax1.hist(no_words_0, color = 'green')
ax1.set_title('No disaster tweets')

no_words_1 = raw_data[raw_data['target'] == 1]['text'].str.split().map(lambda x : len(x))
ax2.hist(no_words_1, color = 'red')
ax2.set_title('Diaster tweets')

plt.suptitle('Number of words in tweets')
plt.show()


# 3. Common stopwords in tweets

# In[44]:


from collections import defaultdict

stop_words = set(nltk.corpus.stopwords.words('english'))

def get_common_words(raw_data,target,stop_words):
  list_of_words=[]
  
  
  
  for x in raw_data[raw_data['target']==target]['text'].str.split():
    for i in x:
      list_of_words.append(i)
  
  dic = defaultdict(int)
  
  for word in list_of_words:
      if word in stop_words:
          dic[word]+=1
          
  top_10 = sorted(dic.items(), key=lambda x:x[1],reverse=True)[:10] 
 
  x,y=zip(*top_10)
  plt.bar(x,y)
  plt.title("List of common words")
  plt.show()
    


# In[45]:


# Common words for No disaster tweets
get_common_words(raw_data,0,stop_words)


# In[46]:


# Common words for Disaster tweets
get_common_words(raw_data,1,stop_words)


# ### Data Cleaning

# In[47]:


data = raw_data.copy()


# #### 1. Remove url's

# In[48]:


def remove_url(text):
  url = re.compile(r"https?://\S+|www\.\S+")
  return url.sub(r"", text)


# In[49]:


data['text'] = data['text'].map(remove_url)


# #### 2. Remove html tags

# In[50]:


def remove_html(text):
  html = re.compile(r'<.*?>')
  return html.sub(r"", text)


# In[51]:


data['text'] = data['text'].map(remove_html)


# #### 3. Remove emojis
# 
# https://stackoverflow.com/questions/33404752/removing-emojis-from-a-string-in-python/49146722#49146722

# In[52]:


def remove_emoji(text):
    
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)


# In[53]:


data['text'] = data['text'].map(remove_emoji)


# #### 4. Remove punctuations

# In[54]:


import  string 
print(string.punctuation)


# In[55]:


def remove_punct(text):
    table = str.maketrans('','',string.punctuation)
    return text.translate(table)


# In[56]:


data['text'] = data['text'].map(remove_punct)


# #### 5. Using GloVe for Vectorization

# In[57]:


from tqdm import tqdm
from nltk.tokenize import word_tokenize
nltk.download('punkt')

def create_corpus(df):
    corpus=[]
    for tweet in tqdm(df['text']):
        words=[word.lower() for word in word_tokenize(tweet) if((word.isalpha()==1) & (word not in stop_words))]
        corpus.append(words)
    return corpus


# In[58]:


corpus = create_corpus(data)


# You can download the below used file 'data_glove.6B.100d.txt' from : https://resources.oreilly.com/conferences/natural-language-processing-with-deep-learning/raw/master/data/glove.6B.100d.txt?inline=false

# In[59]:


embedding_dict={}
with open('/content/drive/MyDrive/Colab Notebooks/Disaster Twitter/data_glove.6B.100d.txt','r') as f:
    for line in f:
        values=line.split()
        word=values[0]
        vectors=np.asarray(values[1:],'float32')
        embedding_dict[word]=vectors


# In[69]:


MAX_LEN=50
tokenizer_obj = tf.keras.preprocessing.text.Tokenizer()
tokenizer_obj.fit_on_texts(corpus)
sequences=tokenizer_obj.texts_to_sequences(corpus)

data_pad = tf.keras.preprocessing.sequence.pad_sequences(sequences,maxlen=MAX_LEN,truncating='post',padding='post')


# In[70]:


word_index = tokenizer_obj.word_index
print('Number of unique words:',len(word_index))


# In[62]:


num_words=len(word_index)+1
embedding_matrix=np.zeros((num_words,100))

for word,i in tqdm(word_index.items()):
    if i > num_words:
        continue
    
    emb_vec=embedding_dict.get(word)
    if emb_vec is not None:
        embedding_matrix[i]=emb_vec


# In[64]:


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Embedding(num_words,100,embeddings_initializer= tf.keras.initializers.Constant(embedding_matrix),
                   input_length=MAX_LEN,trainable=False))
model.add(tf.keras.layers.LSTM(64, dropout=0.1))
model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

model.summary()


# In[66]:


optimzer= tf.keras.optimizers.Adam(learning_rate=1e-5)
model.compile(loss='binary_crossentropy',optimizer=optimzer,metrics=['accuracy'])


# In[72]:


data_pad


# In[73]:


from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(data_pad,data.target.values,test_size=0.15)
print('Shape of train',X_train.shape)
print("Shape of Validation ",X_val.shape)


# In[74]:


history=model.fit(X_train, y_train, batch_size=4, epochs=20, validation_data = (X_val,y_val), verbose=2)


# In[75]:


def plot_results(history):
  training_accuracy = history.history['accuracy']
  validation_accuracy = history.history['val_accuracy']

  plt.plot(training_accuracy, label = 'train_acc')
  plt.plot(validation_accuracy, label = 'val_acc')
  plt.legend()
  plt.xlabel('Epochs')
  plt.ylabel('Accuracy')
  plt.show()

  training_loss = history.history['loss']
  validation_loss = history.history['val_loss']

  plt.plot(training_loss, label = 'train_loss')
  plt.plot(validation_loss, label = 'val_loss')
  plt.legend()
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.show()


# In[76]:


plot_results(history)


# In[77]:


model_json = model.to_json()
with open("tweet.json", "w") as json_file:
  json_file.write(model_json)

# serialize weights to HDF5
model.save_weights("tweet.h5")
print("Saved model to disk !!!")


# Saving Model Architecture

# In[80]:


tf.keras.utils.plot_model(model, "Architecture.png", show_shapes=True)


# In[ ]:





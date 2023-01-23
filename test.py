import re
import numpy as np
import pandas as pd
import jieba
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


#讀取檔案
data = pd.read_excel("article.xlsx")
data.fillna("",inplace=True)
speech_list = list(data["content"])

#讀取字典及停用字
jieba.load_userdict("config/jieba_user_dict.txt")
with open('config/stop_words.txt', encoding = 'UTF-8') as f:
    stop_words = f.readlines()
    
#去除停用字的隱藏格式
stop_words = [w.replace('\n', '') for w in stop_words]  #s.replace(old, new[, max])
stop_words = [w.replace(' ', '') for w in stop_words]

# 去除繁體中文以外的英文、數字、符號
rule = re.compile(r"[^\u4e00-\u9fa5]")
speech_list = [list(jieba.cut(rule.sub('', speech))) for speech in speech_list]
for idx, speech in enumerate(speech_list):
    speech_list[idx] = ' '.join([word for word in speech if word.strip() not in stop_words])


n_topics = 20   ### 分幾個topics
n_top_words = 50   ### 顯示topic中多少個字(關鍵字)

tf_vectorizer = CountVectorizer(token_pattern='[\u4e00-\u9fff]{2,6}',max_features=500)
tf = tf_vectorizer.fit_transform(speech_list)
lda = LatentDirichletAllocation(n_components=n_topics, max_iter=50,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0)
lda.fit(tf)

feature_names = tf_vectorizer.get_feature_names_out()
compoments = lda.components_

#匯出主題關鍵字在excel
topic_dic = {}
for no in range(n_topics):
  topic = ([feature_names[i] for i in compoments[no].argsort()[:-n_top_words - 1:-1]])
  topic_dic["topic"+str(no+1)] = topic

topic_df = pd.DataFrame(topic_dic)
topic_df.to_excel("result.xlsx",index=False)

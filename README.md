# Keyphrase Extraction Algorithm based on Graph

**无监督学习**的中文关键词抽取（Keyphrase Extraction）

- 基于统计：TF-IDF，YAKE

- 基于图：
  - 基于统计：TextRank，SingleRank，SGRank，PositionRank
  - 基于类似文件/引文网络：ExpandRank，CiteTextRank
  - 基于主题：
    - 基于聚类：TopicRank（TR）
  
    - 基于LDA：TPR（TopicPageRank）， Single TPR，Salience Rank
      - 英文Keyphrase Extraction参考：https://github.com/JackHCC/Keyphrase-Extraction
  
  - 基于语义方法：
    - 基于知识图谱：WikiRank
    - 基于预训练词嵌入： theme-weighted PageRank 
  
- 基于嵌入：EmbedRank， Reference Vector Algorithm (RVA)，SIFRank
- 基于语言模型：N-gram

![](./image/Unsupervised Methods.png)

## Introduction

### Statistics-based

| Algorithm |                            Intro                             | Year |                             ref                              |
| :-------: | :----------------------------------------------------------: | :--: | :----------------------------------------------------------: |
|  TF-IDF   | 一种用于信息检索与数据挖掘的常用加权技术，常用于挖掘文章中的关键词 | 1972 |     [link](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)     |
|   YAKE    |     首次将主题（Topic）信息整合到 PageRank 计算的公式中      | 2018 | [paper](https://repositorio.inesctec.pt/server/api/core/bitstreams/ef121a01-a0a6-4be8-945d-3324a58fc944/content) |

### Graph-based

|     Algorithm      |                            Intro                             | Year |                             ref                              |
| :----------------: | :----------------------------------------------------------: | :--: | :----------------------------------------------------------: |
|      TextRank      |           基于统计，将PageRank应用于文本关键词抽取           | 2004 |        [paper](https://aclanthology.org/W04-3252.pdf)        |
|     SingleRank     |       基于统计，TextRank的一个扩展，它将权重合并到边上       | 2008 | [paper](https://www.aaai.org/Papers/AAAI/2008/AAAI08-136.pdf) |
|       SGRank       |              基于统计，利用了统计和单词共现信息              | 2015 |        [paper](https://aclanthology.org/S15-1013.pdf)        |
| PositionRank（PR） |   基于统计，利用了单词-单词共现及其在文本中的相应位置信息    | 2017 |        [paper](https://aclanthology.org/P17-1102.pdf)        |
|     ExpandRank     | 基于类似文件/引文网络，SingleRank扩展，考虑了从相邻文档到目标文档的信息 | 2008 | [paper](https://www.aaai.org/Papers/AAAI/2008/AAAI08-136.pdf) |
|    CiteTextRank    | 基于类似文件/引文网络，通过引文网络找到与目标文档更相关的知识背景 | 2014 | [paper](https://ojs.aaai.org/index.php/AAAI/article/view/8946) |
|  TopicRank（TR）   |     基于主题，使用层次聚集聚类将候选短语分组为单独的主题     | 2013 | [paper](https://hal.archives-ouvertes.fr/hal-00917969/document) |
|        TPR         | 基于主题，首次将主题（Topic）信息整合到 PageRank 计算的公式中 | 2010 |        [paper](https://aclanthology.org/D10-1036.pdf)        |
|     Single TPR     |           基于主题，单词迭代计算的Topic  PageRank            | 2015 | [paper](https://biblio.ugent.be/publication/5974208/file/5974209.pdf) |
|   Salience Rank    |            基于主题，引入显著性的Topic  PageRank             | 2017 |         [paper](https://aclanthology.org/P17-2084/)          |
|      WikiRank      |      基于语义，构建一个语义图，试图将语义与文本联系起来      | 2018 |        [paper](https://arxiv.org/pdf/1803.09000.pdf)         |

### Embedding Based

|            Algorithm             |                            Intro                             | Year |                             ref                              |
| :------------------------------: | :----------------------------------------------------------: | :--: | :----------------------------------------------------------: |
|            EmbedRank             | 使用句子嵌入（Doc2Vec或Sent2vec）在同一高维向量空间中表示候选短语和文档 | 2018 |        [paper](https://arxiv.org/pdf/1801.04470.pdf)         |
| Reference Vector Algorithm (RVA) | 使用局部单词嵌入/语义（Glove），即从考虑中的单个文档中训练的嵌入 | 2018 |        [paper](https://arxiv.org/pdf/1710.07503.pdf)         |
|         SIFRank/SIFRank+         |          基于预训练语言模型的无监督关键词提取新基线          | 2020 | [paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8954611) |

## Dependencies

  - sklearn
  - jieba==0.42.1
  - networkx==2.5
  - numpy==1.20.1
  - pandas==1.2.4
  - matplotlib==3.3.4
  - queue==0.6.3

## File

- `main.py`：主程序入口
- `process.py`：数据预处理和配置加载
- `lda.py`：潜在迪利克雷分配
- `ranks.py`：Topic PageRank算法实现
- `utils.py`：工具函数

## Data

本项目采用新浪新闻8个领域（体育，娱乐，彩票，房产，教育，游戏，科技，股票）的新闻数据共800条作为实验数据。

数据集位于`data/data.xlsx`下，由两列组成，第一列content存放新闻标题和新闻的正文内容，第二列是type是该新闻的话题类型。

在模型训练过程只需要利用excel文件中的`content`列，第二列是根据提取的关键词来衡量提取的准确性。

### 如何使用自己的数据

按照`data.xlsx`的数据格式放置你的数据，只需要`content`列即可。

## Config

`config`目录下可以配置：

- `jieba`分词库的自定义词典`jieba_user_dict.txt`，具体参考：[Jieba](https://github.com/fxsjy/jieba#%E8%BD%BD%E5%85%A5%E8%AF%8D%E5%85%B8)
- 添加停用词（stopwords）`stop_words.txt`
- 添加词性配置`POS_dict.txt`，即设置提取最终关键词的词性筛选，具体词性表参考：[词性表](https://blog.csdn.net/Yellow_python/article/details/83991967)

## Usage

### Install

```shell
git clone https://github.com/JackHCC/Chinese-Keyphrase-Extraction.git

cd Chinese-Keyphrase-Extraction

pip install -r requirements.txt
```

### Run

```shell
# TextRank
python main.py --alg text_rank
# TPR
python main.py --alg tpr
# Single TPR
python main.py --alg single_tpr
# Salience Rank
python main.py
```

### Custom

```shell
python main.py --alg salience_rank --data ./data/data.xlsx --topic_num 10 --top_k 20 --alpha 0.2
```

- `alg`：选择Top PageRank算法，提供四种选择：`text_rank`, `tpr`, `single_tpr`, `salience_rank`
- `data`：训练数据集路径
- `topic_num`：确定潜在迪利克雷分配的主题数量
- `top_k`：每个文档提取关键词的数量
- `alpha`：`salience_rank`算法的超参数，用于控制语料库特异性和话题特异性之间的权衡，取值位于0到1之间，越趋近于1，话题特异性越明显，越趋近于0，语料库特异性越明显

## Result

- TextRank前十条数据提取关键词结果

```
0  :  训练;大雨;球员;队员;队伍;雨水;热身赛;事情;球队;全队;国奥;影响;情况;比赛;伤病
1  :  战术;姑娘;首战;比赛;过程;记者;主帅;交锋;信心;剪辑;将士;软肋;世界杯;夫杯;遭遇
2  :  冠军;活动;女士;文静;游戏;抽奖;俱乐部;眼镜;大奖;特等奖;奖品;现场;环节;教练;球队
3  :  俱乐部;球员;工资;危机;宏运;球队;奖金;管理;老队员;教练;笑里藏刀;前提;集体;集团;经验
4  :  警方;立案侦查;总局;产业;电话;足球;外界;消息;公安部门;依法;中体;主席;裁判;检察机关;委员会
5  :  比赛;鹿队;机会;命中率;队员;联赛;调整;开赛;压力;包袱;外援;主场;状态;体育讯;金隅
6  :  火箭;球队;比赛;原因;时间;效率;开局;事实;教练组;变化;轨道;过程;漫长;判断能力;时机
7  :  胜利;球队;队友;火箭;篮板;比赛;关键;垫底;句式;小牛;新浪;战绩;体育讯;活塞;时间
8  :  火箭;交易;活塞;球队;球员;情况;筹码;价值;命运;市场;续约;掘金;遭遇;球星;核心
9  :  湖人;比赛;球队;后卫;揭幕战;沙农;时间;出场;阵容;板凳;火力;外线;念头;贡献;证明
10  :  公牛;球员;球队;教练;数据;比赛;能力;体系;主教练;命中率;交流;研究;水平;记者;小时
```

- 最终提取结果写入excel表格中，具体在`result`目录下。

## Reference

  - Papagiannopoulou E, Tsoumakas G. A review of keyphrase extraction.

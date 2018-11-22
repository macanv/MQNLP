## MQNLP 一个乱七八糟的项目

这个项目其实在心里想了很多次想开始了，其实也就是上半年的NLP课上开始的，后来项目被git不小心给覆盖掉了。。。
这是一个新的开始，想把最近学的东西都实现了，目标覆盖的东西有:
#### 1.文本分类
- 1. LR
>  采用逻辑回归用于文本分类，其中对TFIDF特征的参数进行了探讨，以及在线性回归的基础上，添加不同的正则项对实验结果造成的影响。同时使用网格方法进行逻辑回归进行调参
http://blog.csdn.net/macanv/article/details/78963762
- 2. naive Bayes
> 采用生成模型朴素贝叶斯进行文本分类，其中探讨了线性调参和n-gram特征分类的性能影响，也使用了网格调参
http://blog.csdn.net/macanv/article/details/78964020
- 3. SVM
> 采用SVM用于文本分类，对不同的核函数在文本上进行了验证，最后发现线性核在文本分类上表现最优，同时由于特征维度巨大，在SVM的计算因为维度灾难造成计算量巨大，借此，引入了降维方法，包括大众的PCA和基于LDA(隐含狄利克雷分配)进行主题的挑选进行降维
- 4. fastText
> 借助Facebook的fasttext api以及基于TensorFlow的fasttext，用于文本分类
- 5. TextCNN
> 基于卷积神经网络对文本进行分类，实验中按照论文：进行了调参实验。
- 6. TextRNN
>基于RNN网络对文本进行分类， 代码只需要提前设置好RNN cell的类型(LSTM,GRU，bi-LSTM, bi-GRU）,同时，指定num_layer,可以轻松创建多层RNN
- 7. TextCNNRNN
> CNN+MaxPooling后接RNN
- 8. ...

####  2. 中文分词
> 包含基于序列标注的BiLSTM-CRF 和一种无须词典的中文分词，基于HMM的中文分词代码后续再实现
> 无须词典的中文分词方法基于粘合度和边界熵的中文分词，方法提出blog:http://www.matrix67.com/blog/archives/5044
   实现地址:https://github.com/Moonshile/ChineseWordSegmentation
> 我在其实现上修改了部分代码，支持python3
##### 2.1 一点废话：
> 无须词典的中文分词，我更倾向于认为他是一个词典生成方法，原始模型不受已经存在的词典影响，直接由训练数据中的写作风格影响，不区分于领域。

#### 3. 命名实体识别
> 添加分词信息的中文命名实体识别，在其基础上进行了修改，论文正在审，录用了会上传最新的代码。
参考: https://github.com/zjy-ucas/ChineseNER

#### 4. 词性标注
> 这部分代码任然使用的是和NER中一样的代码(BiLSTM+CRF 序列标注模型))，没有使用分词特征，在人民日报一月语料库上效果相对会好一些

#### 4. 关系抽取

#### 5. 关键词抽取
> 实现了基于tf-idf、textrank 的关键词抽取
#### 6. 文章摘要
> 正在准备实现基于seq2seq的，苦于没有训练语料

#### 7. GAN学习
> 1. 原始GAN (GAN.py)

> 2. Conditional GAN (cGAN.py)

> 3. 
#### 7. 主题模型(不要face的写上吧)
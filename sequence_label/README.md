## 1.项目简介
大部分代码来自：https://github.com/zjy-ucas/ChineseNER
我在其上面加了些许注释，并且融入了
https://github.com/crownpku/Information-Extraction-Chinese/tree/master/NER_IDCNN_CRF
使用IDCNN替换掉LSTM，加速模型的训练,原始项目中是自定义的RNN Cell,下表是对比使用不同的方法得到的结果:


方法   | F   | 100个epoch运行时间
IDCNN | 89.57  |
自定义Cell| 91.24
LSTM  |  91.41   |     |      |
GRU   |  91.170 |     |      |

可以看出:



## 2. 模型网络介绍
这份代码使用的是RNN_CRF做的命名实体识别任务，网络自底向上分别是
- 1.word 级别的embedding
- 2.char级别的embedding
- 3.两种embedding进行concat ,
- 4.RNN layer
- 5.CRF layer
- 6.Viterbi 解码
- 7.softmax output

### 2.1 实现存在的问题
使用jieba进行word级别的切分，因为jieba的不准确性，给word级别的编码带来了一定干扰，正在考虑更换分词模型或者使用人名日报语料库，这种已经做好了分词的语料。

## 我在其上进行的改进
使用RNN_CNNs_CRF模型，添加了chars级别的convolution pre preocess
代码正在coding.
实验结果对比:


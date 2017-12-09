# coding=utf-8

from keras.models import Sequential
from keras.layers import Dense, Conv1D, Activation, Dropout, GlobalMaxPooling1D, Embedding,Merge
from keras.preprocessing import sequence

class TextCNN():

    def __init__(self, batch_size=32, embedding_mat=None, embedding_dims=200,
                 filters=200, regions_size=[3], hidden_dims=3, epochs=5, keep_dropout_prob=0.5,
                 max_features=10000, maxlen=400, strides=1, padding='valid', actiovation='relu',
                 num_category=2, last_activation='sigmoid',
                 optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']):
        """
        卷积神经网络用于文本分类
        :param batch_size: 
        :param max_features: 最大特征个数 max features size
        :param maxlen: 
        :param embedding_mat: pre_train word2vec or glove 
        :param embedding_dims:  word vector dims
        :param filters: num of filter
        :param region_size: each filter height a list type
        :param hidden_dims: FC hidden dims
        :param epochs: train epoch size 
        """
        self.batch_size = batch_size
        if embedding_mat is not None:
            self.embedding_mat = embedding_mat
            if embedding_mat.shape[1] != embedding_dims:
                raise ValueError('embedding_mat dims not equal embedding_dims')
        self.embedding_dims = embedding_dims
        self.filters = filters
        self.regions_size = regions_size
        self.hidden_dims = hidden_dims
        self.epochs = epochs
        self.keep_dropout_prop = keep_dropout_prob
        self.max_features = max_features
        self.maxlen = maxlen
        self.strides = strides
        self.padding = padding
        self.actiovation = actiovation
        self.num_category = num_category
        self.last_activation = last_activation
        self.optimizer = optimizer
        self.loss =loss
        self.metrics = metrics

        self.model = Sequential()
        self.__build_neuro_network()

    def __build_neuro_network(self):
        """
        网络搭建
        :param max_features: 
        :param maxlen: 
        :param strides: 
        :param padding: 
        :param actiovation: 
        :param num_category: 
        :param last_activation: 
        :return: 
        """
        convs = self.__muliti_region_size_Con1D_and_MaxPooling()
        if len(self.regions_size) > 1:
            self.model.add(Merge(convs, mode='concat'))
        else:
            self.model.add(convs[0])

        # 4. FC
        self.model.add(Dense(self.hidden_dims))
        self.model.add(Dropout(self.keep_dropout_prop))
        self.model.add(Activation('relu'))

        self.model.add(Dense(self.num_category))
        self.model.add(Activation(self.last_activation))

    def __muliti_region_size_Con1D_and_MaxPooling(self):
        """
        实现不同region size的卷积神经网络
        :return: 
        """
        convs = []
        if len(self.regions_size) < 1:
            raise ValueError('region_size must be a list and element size gater 1')

        for region in self.regions_size:
            sub_model = Sequential()
            # 1. embeddig layer
            sub_model.add(Embedding(input_dim=self.max_features,
                                     output_dim=self.embedding_dims,
                                     input_length=self.maxlen))
            sub_model.add(Dropout(self.keep_dropout_prop))

            # 2.convilution layer
            sub_model.add(Conv1D(self.filters, region, strides=self.strides,
                                  padding=self.padding, activation=self.actiovation))

            # 3. pooling
            sub_model.add(GlobalMaxPooling1D())
            convs.append(sub_model)

        return convs

    def train(self, x_train, y_train, x_test, y_test):
        """
        模型训练与评价
        :param optimizer: 
        :param loss: 
        :param merics: 
        :return: 
        """
        self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)

        if len(self.regions_size) > 1:
            x_train *= 3
            x_test *= 3
        self.model.fit(x_train, y_train, batch_size=self.batch_size, epochs=self.epochs,
                       validation_data=(x_test, y_test))

# if __name__ == '__main__':
#     from keras.datasets import imdb
#     from keras.preprocessing import sequence
#
#     (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=5000, maxlen=400)
#     x_train = sequence.pad_sequences(x_train, 400)
#     x_test = sequence.pad_sequences(x_test, 400)
#
#     clf = TextCNN(epochs=3, batch_size=32, embedding_dims=50, filters=250, regions_size=[3, 4, 5], hidden_dims=250, num_category=1, last_activation='sigmoid')
#     clf.train(x_train, y_train, x_test, y_test)


import keras
import numpy as np
from keras.preprocessing import sequence
from models.keras_impl.text_classification import TextCNN
from utils import thu
from sklearn.cross_validation import train_test_split

if __name__ == '__main__':
    input_x, input_y, words_index = thu.load_data_from_file(file_path='/')
    x_train, x_test, y_train, y_test = train_test_split(input_x,
                                                        input_y,
                                                        train_size=0.9, random_state=123)
    maxlen = 400
    # x_train = sequence.pad_sequences(x_train, maxlen)
    # x_test = sequence.pad_sequences(x_test, maxlen)

    y_train = keras.utils.to_categorical(y_train, num_classes=3)
    y_test = keras.utils.to_categorical(y_test, num_classes=3)
    clf = TextCNN(num_category=3, maxlen=maxlen, epochs=20)
    clf.train(x_train, y_train, x_test, y_test)

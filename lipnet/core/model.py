from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, TimeDistributed, Flatten
from tensorflow.keras.layers import Dense, Activation, Dropout, Bidirectional, GRU
from lipnet.core.layers import CTC

class LipNet:
    def __init__(self, img_c, img_w, img_h, output_size, absolute_max_string_len, max_string_len):
        self.img_c = img_c
        self.img_w = img_w
        self.img_h = img_h
        self.output_size = output_size
        self.absolute_max_string_len = absolute_max_string_len
        self.max_string_len = max_string_len
        self.build()

    def build(self):
        input_data = Input(name='the_input',
                           shape=(self.img_w, self.img_h, self.img_c))

        # 3D Convolution
        x = Conv3D(32, kernel_size=(3, 5, 5), padding='same',
                   activation='relu')(input_data)
        x = MaxPooling3D(pool_size=(1, 2, 2))(x)

        x = Conv3D(64, kernel_size=(3, 5, 5), padding='same',
                   activation='relu')(x)
        x = MaxPooling3D(pool_size=(1, 2, 2))(x)

        x = Conv3D(96, kernel_size=(3, 3, 3), padding='same',
                   activation='relu')(x)
        x = MaxPooling3D(pool_size=(1, 2, 2))(x)

        # Reshape for RNN
        x = TimeDistributed(Flatten())(x)

        # RNN
        x = Bidirectional(GRU(256, return_sequences=True, dropout=0.5))(x)
        x = Bidirectional(GRU(256, return_sequences=True, dropout=0.5))(x)

        # FC + Softmax
        x = Dense(self.output_size, kernel_initializer='he_normal')(x)
        y_pred = Activation('softmax', name='y_pred')(x)

        # Model for training
        labels = Input(name='the_labels', shape=[self.max_string_len], dtype='float32')
        input_length = Input(name='input_length', shape=[1], dtype='int64')
        label_length = Input(name='label_length', shape=[1], dtype='int64')
        loss_out = CTC(name='ctc')([y_pred, labels, input_length, label_length])

        self.model = Model(inputs=[input_data, labels, input_length, label_length],
                           outputs=loss_out)

        # Model for prediction
        self.model_pred = Model(inputs=input_data, outputs=y_pred)

import tensorflow.keras as keras
from tensorflow.keras.layers import Conv2D, Dense, Flatten


class DuelingDeepQNetwork(keras.Model):
    def __init__(self, input_dims, n_actions):
        super(DuelingDeepQNetwork, self).__init__()
        self.conv1 = Conv2D(32, 8, strides=(4, 4), activation='relu',
                            data_format='channels_first',
                            input_shape=input_dims)
        self.conv2 = Conv2D(64, 4, strides=(2, 2), activation='relu',
                            data_format='channels_first')
        self.conv3 = Conv2D(64, 3, strides=(1, 1), activation='relu',
                            data_format='channels_first')
        self.flat = Flatten()
        self.fc1 = Dense(512, activation='relu')
        self.A = Dense(n_actions, activation=None)
        self.V = Dense(1, activation=None)

    def call(self, state):
        x = self.conv1(state)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flat(x)
        x = self.fc1(x)
        V = self.V(x)
        A = self.A(x)

        return V, A

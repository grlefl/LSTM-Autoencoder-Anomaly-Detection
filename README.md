# LSTM Autoencoder for Anomaly Detection in Sequential Data

## Table of Contents
- [Project Status](#project-status)
- [Overview](#overview)
- [User Documentation](#user-documentation) (needs updating)
- [Dataset](#dataset)
- [Developer Documentation](#developer-documentation)
- [Results](#results) (needs updating)
- [Presentation](#presentation)
- [Next Steps](#next-steps) 

## Project Status 
This project has not achieved viable results. [Next Steps](#next-steps) are listed below.

## Overview 
This program attempts to use an LSTM autoencoder to detect anomalies the daily closing prices of the S&P 500 index. The autoencoder is trained exclusively on normal data without anomalies. When new data points are introduced to the model, they are classified as anomalies if they exceed a predefined threshold.

## User Documentation
(needs updating) 

## Dataset 
The dataset includes the daily closing prices of the S&P 500 index from 1986 to 2018. It is provided by [Patrick David](https://twitter.com/pdquant) and hosted on [Kaggle](https://www.kaggle.com/datasets/pdquant/sp500-daily-19862018). The data contains only two columns/features: the date and the closing price.

## Developer Documentation 
This project was originally inspired by an [article](https://curiousily.com/posts/anomaly-detection-in-time-series-with-lstms-using-keras-in-python/) and [github repository](https://github.com/lestercardoz11/SP-500-index-anomaly-detection) where the LSTM autoencoder is a Keras model. My goal was to gain a better understanding of the LSTM autoencoder structure by achieving similar anomaly detection results with a Pytorch implemention.

Keras Implementation (from [article](https://curiousily.com/posts/anomaly-detection-in-time-series-with-lstms-using-keras-in-python/))

```
model = keras.Sequential()
model.add(keras.layers.LSTM(
    units=64,
    input_shape=(X_train.shape[1], X_train.shape[2])
))
model.add(keras.layers.Dropout(rate=0.2))
model.add(keras.layers.RepeatVector(n=X_train.shape[1]))
model.add(keras.layers.LSTM(units=64, return_sequences=True))
model.add(keras.layers.Dropout(rate=0.2))
model.add(
  keras.layers.TimeDistributed(
    keras.layers.Dense(units=X_train.shape[2])
  )
)
model.compile(loss='mae', optimizer='adam')
```

My Pytorch Implementation 

```
class Encoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=64):
        super(Encoder, self).__init__()
        self.seq_len, self.n_features = seq_len, n_features
        self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim

        # first LSTM layer
        self.rnn1 = nn.LSTM(
            input_size=n_features,
            hidden_size=self.hidden_dim,
            batch_first=True
        )
        # initializing the hidden numbers of layers
        self.rnn2 = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=embedding_dim,
            batch_first=True
        )

    def forward(self, x):
        x, (hidden_n, _) = self.rnn1(x)  # input (batch, seq_len, n_features)
        x, (_, _) = self.rnn2(x)  # hidden state is input for next layer, output last layer of LSTM
        return x
```

```
class Decoder(nn.Module):
    def __init__(self, seq_len, input_dim=64, n_features=1):
        super(Decoder, self).__init__()
        self.seq_len, self.input_dim = seq_len, input_dim
        self.hidden_dim, self.n_features = 2 * input_dim, n_features

        # first LSTM layer
        self.rnn1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=input_dim,
            batch_first=True
        )
        # using a dense layer as an output layer
        self.rnn2 = nn.LSTM(
            input_size=input_dim,
            hidden_size=self.hidden_dim,
            batch_first=True
        )

        self.output_layer = nn.Linear(self.hidden_dim, n_features)  # reduce features to match input

    def forward(self, x):
        x, (hidden_n, _) = self.rnn1(x)
        x, (_, _) = self.rnn2(x)  # hidden state is input for next layer, output last layer of LSTM
        return self.output_layer(x)
```

The different parts of the program include data prep, something and etc. 

## Results 
(needs updating) See [Presentation](#presentation).

## Presentation
(youtube video not yet available)

![image](https://github.com/grlefl/Phase-2/assets/124198528/79731d33-489f-4cf9-bacc-ca373c3c21fa)
![image](https://github.com/grlefl/Phase-2/assets/124198528/299d41e9-20c0-4a9e-91ac-bd32c78ad074)
![image](https://github.com/grlefl/Phase-2/assets/124198528/84096516-9e41-4306-8a1a-2764a50bee6e)
![image](https://github.com/grlefl/Phase-2/assets/124198528/f8a025cc-7dee-4a89-9d2e-cd2cab181225)
![image](https://github.com/grlefl/Phase-2/assets/124198528/c295142a-ecb6-464a-a892-8a56d34b42b8)
![image](https://github.com/grlefl/Phase-2/assets/124198528/54326fbd-8863-4ead-942b-a0df412488ad)
![image](https://github.com/grlefl/Phase-2/assets/124198528/9c79745f-5866-4c5a-8730-05d3421da451)
![image](https://github.com/grlefl/Phase-2/assets/124198528/ac3c111c-f747-4d1d-8c2d-dcd6d32ab131)
![image](https://github.com/grlefl/Phase-2/assets/124198528/1c7ae360-d4d2-4c1c-92bf-95826ab84f2d)

## Next Steps 
-- TO BE COMPLETED -- 

- implement the keras model to see if I can get some baseline results
- keep working on the LSTM pytorch implementation, include dropouts etc 

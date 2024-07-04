# LSTM Autoencoder for Anomaly Detection in Sequential Data

## Table of Contents
- [Project Status](#project-status)
- [Overview](#overview)
- [User Documentation](#user-documentation)
- [Dataset](#dataset)
- [Developer Documentation](#developer-documentation)
- [Results](#results) 
- [Presentation](#presentation)
- [Next Steps](#next-steps) 

## Project Status 
This project has not achieved viable results. [Next Steps](#next-steps) are listed below.

## Overview 
This program attempts to use an LSTM autoencoder to detect anomalies the daily closing prices of the S&P 500 index. The autoencoder is trained exclusively on normal data without anomalies. When new data points are introduced to the model, they are classified as anomalies if they exceed a predefined threshold.

## User Documentation
(incomplete) 

## Dataset 
The dataset that is used is the daily closing prices of the S&P 500 index from 1986 to 2018. It is provided by [Patrick David](https://twitter.com/pdquant) and hosted on [Kaggle](https://www.kaggle.com/datasets/pdquant/sp500-daily-19862018). The data contains only two columns/features: the date and the closing price.

## Developer Documentation 
This project is originally inspired by this article and github repository where a keras model is used for the LSTM autoencoder. My goal was to basically do this same project but with a PyTorch model to gain a deeper understanding of the autoencoder structure. The keras model is shown below, and here is the current structure of my pytorch autoencoder. 
Keras Implementation 
![image](https://github.com/grlefl/LSTM-Autoencoder-SP500/assets/124198528/272eaac1-af0c-47ae-a338-b900e2d188d6)

My Pytorch Implementation 
![image](https://github.com/grlefl/LSTM-Autoencoder-SP500/assets/124198528/e863a33f-4e3c-42ec-8c67-3d538ec0ba3a)
![image](https://github.com/grlefl/LSTM-Autoencoder-SP500/assets/124198528/f6bb42a1-b280-4a5b-b942-51bb04e92faa)

The different parts of the program include data prep, something and etc. 

## Results 
The results for this project are not very good. Here they are: 
Initially I thought the data preparation was wrong, but I think the overall structure of the LSTM autoencoder is wrong. This article gives a really good indepth explanation about how the dimensions should be. There is dropout stuff that I forgot to add and whatever. 

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
- implement the keras model to see if I can get some baseline results
- keep working on the LSTM pytorch implementation, include dropouts etc 

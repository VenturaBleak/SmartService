
## Table of Contents

1. [Introduction](#introduction)
2. [Session Data Cleaning](#session-data-cleaning)
3. [Feature Gathering and Cleaning](#feature-gathering-and-cleaning)
4. [Outlier Detection](#outlier-detection)
5. [Final Data Preparation](#final-data-preparation)
6. [Model Data](#model-data)

## Introduction <a name="introduction"></a>

This repository holds the code for our smart service solution. This README file provides the name, purpose, and usage of each uploaded code in the respective folders. Please make sure to install all necessary packages before running these codes.

This document follows the steps we took when approaching our solution, beginning with cleaning the charging session data and concluding with running the model. For a complementary dashboard solution please contact the contributors.

## Session Data Cleaning <a name="session-data-cleaning"></a>

`1.session_data_cleaning` - This folder contains scripts for functions for cleaning the charging session data. It's the first step in our pipeline and prepares the raw data for feature gathering.

## Feature Gathering and Cleaning <a name="feature-gathering-and-cleaning"></a>

`2.feature_gathering_cleaning` - This folder contains scripts that extract useful features from various sources about the zipcodes of the city.

## Outlier Detection <a name="outlier-detection"></a>

`3.outlier_detection` - This folder contains scripts that identifies and deals with any outliers in the dataset to ensure the robustness and accuracy of the subsequent model training and prediction.

## Final Data Preparation <a name="final-data-preparation"></a>

`4.final_data_preparation` - This folder contains scripts that are used to finalize the dataset before model training, including tasks such as scaling and normalization.

## Model Data <a name="model-data"></a>

`5.model_data` - Contains various scripts for preparing and running different models.
---

For more information on each script, please refer to the comments within the files. Happy coding! :) 

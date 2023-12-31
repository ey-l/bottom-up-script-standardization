# Bottom-Up Standardization for Data Preparation

## Data

Experiment datasets: https://github.com/ey-l/bottom-up-script-standardization/blob/main/data.zip

This file contains the six Kaggle competitions we crawled and cleaned.
* House Prices - Advanced Regression Techniques: https://www.kaggle.com/c/house-prices-advanced-regression-techniques
* Predict Future Sales: https://www.kaggle.com/c/competitive-data-science-predict-future-sales
* Titanic - Machine Learning from Disaster: https://www.kaggle.com/competitions/titanic
* Spaceship Titanic: https://www.kaggle.com/competitions/spaceship-titanic
* Natural Language Processing with Disaster Tweets: https://www.kaggle.com/competitions/nlp-getting-started
* Pima Indians Diabetes Database: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database

## Environment

Virtual env files: https://github.com/ey-l/bottom-up-script-standardization/tree/main/exp-env

``` conda env create -f lucid_env.yml -- lucid ```

## Prototype System

The system has the following components:
* Translating an AST to a graph: https://github.com/ey-l/bottom-up-script-standardization/blob/main/lucidscript/ASTDAG.py
* Translating a graph to our DAG representation: https://github.com/ey-l/bottom-up-script-standardization/blob/main/lucidscript/LUCIDDAG.py
* Our search framework: https://github.com/ey-l/bottom-up-script-standardization/blob/main/lucidscript/LUCID.py
* User intent estimation: https://github.com/ey-l/bottom-up-script-standardization/blob/main/lucidscript/correctness.py
* Utils: https://github.com/ey-l/bottom-up-script-standardization/blob/main/lucidscript/utils.py

## GPT experiments

GPT survey and student responses: https://github.com/ey-l/bottom-up-script-standardization/blob/main/GPT-prompt-survey.md

GPT experiment script: https://github.com/ey-l/bottom-up-script-standardization/tree/main/llms
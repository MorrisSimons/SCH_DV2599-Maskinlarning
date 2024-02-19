# Project Title: Predicting Song Popularity with Spotify Data

## Overview
This project aims to predict the popularity of songs based on Spotify data using machine learning techniques. We explore various audio features provided by Spotify, such as duration, key, mode, and time signature, to understand their impact on a song's popularity.

## Tips
Before reviewing the notebook, make sure all cells are collapsed. This will make it easier to navigate the notebook and understand the flow.

## Dataset
The dataset includes a wide range of songs from Spotify, with features like `duration_ms`, `instrumentalness`, and `mode` among others. Songs are classified based on their popularity, which is initially a continuous score from 0 to 100. To streamline our analysis, we focus on binary classification by labeling songs as "hits" or "misses."

The dataset can be found on: https://www.kaggle.com/datasets/zaheenhamidani/ultimate-spotify-tracks-db

## Data Preprocessing
- **Duration Filtering**: We removed songs longer than 500,000 milliseconds (8 minutes and 20 seconds) and shorter than 60,000 milliseconds (1 minute) to focus on tracks within a standard length range.
- **Feature Encoding**: Non-integer attributes like `key` and `mode` were converted to numerical formats. The `key` attribute was mapped to a 0-11 scale, while major and minor modes were coded as 1 and 0, respectively.

## Methodology
The analysis involves several machine learning models to predict song popularity. The preprocessing steps include normalizing features and handling categorical data to prepare the dataset for modeling.

## Key Findings
The Accuracy of the models was arount `dsdssdsd`% for out hit or miss classification. The most important features for the models were `dsdsds`, `sddssdsd`, and `dsdssdsd`. The Random Forest model outperformed the other models with an accuracy of ""%.

## How to Use
1. **Installation**: Ensure you have Jupyter Notebook or JupyterLab installed.
2. **Dependencies**: Install required Python packages listed in `requirements.txt` with `pip install -r requirements.txt`.

3. **Running the Notebook**: Open `main.ipynb` in Jupyter and execute the cells sequentially.

## Contributors
-Isak Rlander, isru21@student.bth.se
-Morris Simons, mosi21@student.bth.se

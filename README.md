# Powerlifting & Weightlifting Performance Analysis with DOTS and Wilks Scores

## Overview

This project is a **Streamlit-based web application** that visualizes and analyzes powerlifting and weightlifting data using **DOTS** and **Wilks** scores. The app enables users to explore and compare the performance of lifters across different categories such as gender, weight class, and equipment. By leveraging machine learning and polynomial regression, the application provides insights into the relationship between body weight and performance for key lifts in the sports of Powerlifting, Olympic Lifting and Strongman.

## Features

- **Interactive Visualizations**: Users can interactively explore data using:
  - **Box plots** for comparing DOTS and Wilks scores across sex, weight classes, and equipment types.
  - **Polynomial regression plots** to visualize the relationship between body weight and performance for different lifts.
  - **Downsampled data** to improve performance and responsiveness, especially with large datasets.
  
- **Lifter Performance Comparison**:
  - Compare **DOTS** scores (adjusted for body weight) across sex, weight classes, and equipment types.
  - Compare **Wilks** scores to assess relative performance in powerlifting.
  
- **Polynomial Regression**:
  - Apply polynomial regression to visualize trends in the relationship between **body weight** and performance in **Squat**, **Bench**, **Deadlift**, **Snatch**, and **Clean & Jerk**.
  - Interactive **polynomial degree** selector to adjust the complexity of the regression line.

### Data and How It Works


## Data

The project uses two datasets:
1. **Powerlifting Data** (`sampled_openpowerlifting.csv`): Contains information on lifters' bodyweight, best lifts (Squat, Bench, Deadlift), and Wilks scores. From https://www.kaggle.com/datasets/open-powerlifting/powerlifting-database/versions/1/data?select=openpowerlifting.csv
2. **Weightlifting Data** (`weightlifting.csv`): Contains information on olympic bodyweight, **Snatch**, and **Clean & Jerk** lifts. From https://www.kaggle.com/datasets/yuxinc/summer-olympics-weightlifting-records-2000-to-2020


### Columns Used:

- **Powerlifting**:
  - `BodyweightKg`, `Best3SquatKg`, `Best3BenchKg`, `Best3DeadliftKg`, `Wilks`, `Sex`, `Equipment`, `WeightClassKg`
  
- **Weightlifting**:
  - `Bodyweight_(kg)`, `Snatch_(kg)`, `Clean_&_Jerk_(kg)`

## How It Works

1. **DOTS and Wilks Scores**:
   - **DOTS Score**: A formula that adjusts a lifter's total based on body weight, enabling comparison across weight classes.
   - **Wilks Score**: A widely-used coefficient that accounts for bodyweight in powerlifting, allowing lifters of different weights to compare their totals.

2. **Downsampling**:
   - To optimize performance, the app downsampled datasets before visualization, retaining key points like minimum, maximum, and representative points in between.

3. **Polynomial Regression**:
   - Polynomial regression is applied to explore the relationship between bodyweight and performance across various lifts. Users can adjust the degree of the polynomial to control the complexity of the regression line.

## Key Functions

- **downsample_data()**: Reduces the dataset to include key points (min, max, and every 1/10th point) to improve performance.
- **poly_reg()**: Performs polynomial regression and returns the regression line and R² score.
- **plot_poly()**: Creates a scatter plot with a polynomial regression line overlaid.


## Acknowledgments

- Data sources from **Kaggle** and **OpenPowerlifting**.
- This README was generated with help from ChatGPT 4o on 10/17.




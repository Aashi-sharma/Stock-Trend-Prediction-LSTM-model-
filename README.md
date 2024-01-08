# Stock-Trend-Prediction-LSTM-model-
Stock Trend  Prediction (LSTM model)
# Stock Trend Prediction Web App

## Overview
This web app predicts stock trends using a neural network model trained on historical stock data. Users can input a stock ticker, and the app provides visualizations of historical data, moving averages, and predictions.

## Dependencies
List all the external libraries and dependencies your project relies on. Include version numbers if necessary.

Python 3.x
NumPy
Pandas
Matplotlib
TensorFlow
Streamlit

## Features
- Visualizations of closing prices over time
- Moving averages (100-day and 200-day)
- Prediction of stock prices using a neural network model
- Streamlit-powered interactive web application
- Based on LSTM model

## LSTM model
Long Short-Term Memory (LSTM) is a type of recurrent neural network (RNN) architecture designed to capture and learn patterns in sequential data. In the context of stock trend prediction, LSTM is used to model the time-dependent relationships within historical stock prices, allowing the algorithm to make predictions based on patterns learned from past data.

## Usage
1. Install the required dependencies by running:
    ```bash
    pip install -r requirements.txt
    ```

2. Activate your virtual environment:
    ```bash
    # On Windows
    .\virtual_environment_name\Scripts\activate

    # On Unix/Linux or macOS
    source virtual_environment_name/bin/activate
    ```

3. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```

4. Open your browser and navigate to the provided URL to use the app.

## Streamlit Apllication
Streamlit is an open-source Python library that simplifies the process of creating web applications for data science and machine learning. It allows developers and data scientists to turn data scripts into interactive web applications quickly and with minimal effort. Streamlit is designed to be easy to use, even for those without extensive web development experience.

Key features of Streamlit include:

Rapid Prototyping: Streamlit provides a simple Python script-based approach to creating web applications, allowing developers to prototype and iterate quickly.

Widgets and Interactivity: Users can easily add interactive widgets (like sliders, buttons, and text inputs) to manipulate data and visualize changes in real-time.

Data Visualization: Streamlit integrates seamlessly with popular data visualization libraries, such as Matplotlib, Plotly, and Altair, making it easy to create charts and graphs.

Machine Learning Integration: Streamlit is commonly used for showcasing machine learning models and results. It supports integration with popular machine learning frameworks like TensorFlow and PyTorch.

Built-in Deployment: Streamlit has built-in support for deployment to various platforms, making it easy to share web applications with others.

## Streamlit Application
The user interface is built using [Streamlit](https://streamlit.io/), a Python library for creating web applications with minimal effort. Streamlit provides an interactive and user-friendly way to visualize and explore the stock trend predictions.

## Data Source
The historical stock data is obtained using the Yahoo Finance API through the `yfinance` library.

## Model
The predictive model is a neural network trained on historical stock data. The model is loaded from the `keras_model.h5` file.

## Author
[Aashi Sharma]

## License
This project is licensed under the [MIT License](LICENSE).



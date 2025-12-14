# 📈 Walmart Store Sales Forecasting (Synthetic Data)

## Project Overview
This project is a Time Series Forecasting application built in Python. It predicts future weekly sales for a retail store. 

The project utilizes **Facebook Prophet**, a robust forecasting library, to model trends and seasonality (e.g., holiday spikes). Due to data access restrictions, this project includes a **Data Simulation Module** that generates synthetic sales data with realistic statistical properties (upward trends, sine-wave seasonality, and random noise).

## 🎯 Objectives
* Generate a synthetic time-series dataset that mimics retail behavior.
* Implement the **Prophet** algorithm to model weekly sales.
* Visualize the "Actual vs. Predicted" sales using Matplotlib.
* Demonstrate understanding of Trend, Seasonality, and Forecasting horizons.

## 🛠️ Tech Stack
* **Language:** Python 3.x
* **Libraries:**
    * `pandas` (Data manipulation)
    * `numpy` (Math & Data generation)
    * `prophet` (Forecasting model)
    * `matplotlib` (Visualization)

## ⚙️ Installation

1. Clone this repository or download the files.
2. Install the required Python packages:

```bash
pip install pandas numpy prophet matplotlib
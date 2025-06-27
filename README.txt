This project analyzes Tesla (TSLA) stock price using Python.

Features
* Data from yFinance (2022–2024)
* Moving Averages: 20-day & 50-day
* Linear Regression for trend prediction
* Crossover detection: Golden & Death Cross
* Visualizations with Matplotlib

Tools Used
* Python
* Pandas
* yFinance
* Matplotlib
* Scikit-learn

Project Overview
This mini?project demonstrates how to download, process, and visualise real?world financial data in Python. 
We analyse Tesla?s stock (ticker *TSLA*) over three years, calculate short? and medium?term moving
averages, locate *Golden?Cross* and *Death?Cross* events, and fit a simple linear?regression trend line.

Objectives
1. Build a compact yet convincing example of data?driven equity analysis. 
2. Showcase core skills: data sourcing, feature engineering, technical?indicator logic, charting, and basic
modelling. 

conclusions 
Moving averages remain a fast, intuitive filter for regime detection.
* Golden/Death Cross logic can act as a first?pass trading signal but should be stress?tested with
commissions and slippage.
* The linear model is deliberately simple; richer models (e.g. tree?based or LSTM) could capture non?linear
dynamics.

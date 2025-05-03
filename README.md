# Honors_College_Capstone
Rutgers University Honors College Capstone Project involving optimizing rowing boat speed
Author: Katharina Dowlin  

This project investigates the factors influencing boat speed in collegiate rowing using stroke-level and boat-level sensor data. The goal is to identify key metrics that optimize performance and improve synchronization in crew rowing through statistical modeling and machine learning.  

## Objectives  
- Quantify the impact of stroke synchronization, power consistency, and technical form on boat speed  
- Apply mixed-effects models to control for race-level variability  
- Build predictive models using supervised machine learning to forecast 500m split times  
- Provide actionable insights for athlete training and race strategy  

## Data  
Data was collected using **Empower Oarlock sensors** during competitive regattas. Features include:  
- Stroke rate  
- Effective length  
- Catch and finish angles  
- Power per stroke  
- Synchronization variability metrics (e.g., SD of catch/finish timing, desync index)  
- Boat velocity (split time over 500m)  

*Note: Raw data is excluded from this repo for privacy reasons.*

## Methods  
- **Exploratory Data Analysis (EDA)** using `pandas`, `matplotlib`, and `seaborn`  
- **Mixed-Effects Linear Models** with `statsmodels` to account for race as a random effect  
- **Machine Learning Models**:
  - Random Forest Regressor     

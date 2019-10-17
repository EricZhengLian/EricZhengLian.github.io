---
layout: post
title: Regression Project
---

This Regression Project did a series of analysis on the dataset of Happiness Scores of 157 different countries around the world in 2016. I picked the Happiness Scores as the response variable in my model. After seeing the bivariate associations between the response variable and all the other candidate explanatory variables, I decided to apply a multiple, RidgeCV linear regression model to my dataset. Before creating the model, I transformed one of the explanatory variables to its cube value to better achieve the assumption of linear correlation of multiple linear regression. Finally, I split the data into 70% : 30% and use the first 70% as my training/validation set and the rest as my test set. The outcomes appear to be pretty accurate and have good variability. You can find my code located [here](https://github.com/EricZhengLian/Regression_Project)

<img src="/images/happy.png" width="600"/>
 
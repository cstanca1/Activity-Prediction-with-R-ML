## Coursera - Practical Machine Learning - Prediction Assignment Write-up

```
file name: README.md
date: 1/31/2016
author: cstanca1
```

### Introduction

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self-movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. 

More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

### Problem

People regularly quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants which were asked to perform barbell lifts correctly and incorrectly in 5 different ways. 

### Data

```
# Training data: https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv
# Test data: https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv
# The data for this project came from: http://groupware.les.inf.puc-rio.br/har
```

### Objective

The objective of this project was to used the data collected during the exercises of those 6 participants to predict the manner in which they did the exercise.

### Output

A report describing how the model was built (PAW.html), how cross validation was used, expected out of sample error is, and the choices made. The R markdown (PAW.Rmd) and the compiled HTML report (PAW.md) describing the analysis were posted to the current Github repo. The repo a link was provided for the Peer Review. 

Note:

The text of the writeup, as required, was constrained to less than 2000 words and the number of figures to less than 5. The repo was submitted with a gh-pages branch so the HTML page can be viewed online, for an easy review.

### Reproducibility

Due to security concerns with the exchange of R code, Coursera recommends that the code will not be run during the evaluation, instead, the HTML version of the analysis will be used for review.

### Course Project Prediction Quiz Portion

The machine learning algorithm developed was also applied to 20 test cases available in the test data and the predictions were submitted in appropriate format to the Course Project Prediction Quiz for automated grading.

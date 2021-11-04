# Mental-Health-Study-in-Tech-Space
Everybody knows that mental health of employees has the potential to have a direct impact on business and revenue.
With this project, we analyze which factors influence mental health the most.
We also recommend how can an organization contribute towards the mental well-being of employees.

# Data Source
Our data source is a survey of employees' mental wellbeing. It measures attitudes towards mental health and frequency of mental health disorders in the tech workplace.
It has 27 variables with 1259 observations.

# Data Preprocessing
We started with reading data in csv format and storing in data frame. We removed variables which we are not considering in our implementation. As the data is taken from a web survey, many records of the data require cleaning to get a meaningful definition. As most of the data is categorized, we converted them into categorical variables.
We partitioned our data into train(60%) and test(40%) sets.

# Data Modeling
We fitted the data to 6 data models and also applied ensemble model to increase accuracy. Also we compared the accuracy of model with the help of Confusion Matrix.

# Conclusion
We concluded with the predictors which have high importance in terms of employees mental wellbeing at work place.


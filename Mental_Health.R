rm(list = ls())
library(dplyr)
library(ggplot2)
library(plotly)
library(reshape2)
library(magrittr)
library(ggthemes)
library(tidyr)
library(DT)
library(lubridate)
library(stringr)
library(RColorBrewer)
library(dplyr)
library(rpart)
library(rpart.plot)
library(party)
library(rattle)
library(partykit)
library(caret)
library(randomForest)
library(xgboost)
library(rpart.plot)
library(VIM)
library(class)
library(randomForest)
library(FastKNN)
library(caTools)
library(glmnet)
library(gbm)
library(kknn)
library(devtools)
library(adabag)
library(rpart) 
library(caret)
library(rpart)
library(MASS)
library(TH.data)
library(neuralnet)
library(randomForestExplainer)
library(forecast)
library(caret)
library(e1071)
library(naivebayes)
library(dplyr)
library(ggplot2)
library(psych)
library(pROC)


setwd("C:/Users/shail/OneDrive - Georgia State University/Desktop/College_Documents/Study/Data Analytics/ClassWork")
data <- read.csv('survey.csv', stringsAsFactors = TRUE)
# To delete no important elements
data <- data[ , !(names(data) %in% "state")]
data <- data[ , !(names(data) %in% "Timestamp")]
data <- data[ , !(names(data) %in% "comments")]
data <- data[ , !(names(data) %in% "self_employed")]


# Gender unification.
data$Gender %<>% str_to_lower()

male_str <- c("male", "m", "male-ish", "maile", "mal", "male (cis)", "make", "male ", "man","msle", "mail", "malr","cis man", "cis male")
trans_str <- c("trans-female", "something kinda male?", "queer/she/they", "non-binary","nah", "all", "enby", "fluid", "genderqueer", "androgyne", "agender", "male leaning androgynous", "guy (-ish) ^_^", "trans woman", "neuter", "female (trans)", "queer", "ostensibly male, unsure what that really means" )
female_str <- c("cis female", "f", "female", "woman",  "femake", "female ","cis-female/femme", "female (cis)", "femail")
data$Gender <- sapply(as.vector(data$Gender), function(x) if(x %in% male_str) "male" else x )
data$Gender <- sapply(as.vector(data$Gender), function(x) if(x %in% female_str) "female" else x )
data$Gender <- sapply(as.vector(data$Gender), function(x) if(x %in% trans_str) "trans" else x )
data %<>% filter(Gender != "a little about you")
data %<>% filter(Gender != "guy (-ish) ^_^")
data %<>% filter(Gender != "p")

# creating categorical variables

data$Gender <- factor(data$Gender, levels = c("trans", "female", "male"), 
                            labels = c(0,1,2))
data$family_history <- factor(data$family_history, levels = c("No","Yes"), 
                      labels = c(0,1))
data$tech_company <- factor(data$tech_company, levels = c("No","Yes"), 
                              labels = c(0,1))
data$remote_work <- factor(data$remote_work, levels = c("No","Yes"), 
                              labels = c(0,1))
data$treatment <- factor(data$treatment, levels = c("No","Yes"), 
                              labels = c(0,1))
data$obs_consequence <- factor(data$obs_consequence, levels = c("No","Yes"), 
                              labels = c(0,1))
data$work_interfere <- factor(data$work_interfere, levels = c("Never", "Rarely", "Often","Sometimes"), 
                      labels = c(0,1,2,3))
data$supervisor <- factor(data$supervisor, levels = c("No", "Yes", "Some of them"), 
                      labels = c(0,1,2))
data$benefits <- factor(data$benefits, levels = c("No", "Yes", "Don't know"), 
                          labels = c(0,1,2))
data$wellness_program <- factor(data$wellness_program, levels = c("No", "Yes", "Don't know"), 
                        labels = c(0,1,2))
data$seek_help <- factor(data$seek_help, levels = c("No", "Yes", "Don't know"), 
                        labels = c(0,1,2))
data$anonymity <- factor(data$anonymity, levels = c("No", "Yes", "Don't know"), 
                        labels = c(0,1,2))
data$mental_vs_physical <- factor(data$mental_vs_physical, levels = c("No", "Yes", "Don't know"), 
                         labels = c(0,1,2))
data$care_options <- factor(data$care_options, levels = c("No", "Yes", "Not sure"), 
                          labels = c(0,1,2))
data$coworkers <- factor(data$coworkers, levels = c("No", "Yes", "Some of them"), 
                          labels = c(0,1,2))
data$mental_health_consequence <- factor(data$mental_health_consequence, levels = c("No", "Yes", "Maybe"), 
                          labels = c(0,1,2))
data$phys_health_consequence <- factor(data$phys_health_consequence, levels = c("No", "Yes", "Maybe"), 
                                         labels = c(0,1,2))
data$mental_health_interview <- factor(data$mental_health_interview, levels = c("No", "Yes", "Maybe"), 
                                         labels = c(0,1,2))
data$phys_health_interview <- factor(data$phys_health_interview, levels = c("No", "Yes", "Maybe"), 
                                       labels = c(0,1,2))
data$leave <- factor(data$leave, levels = c("Don't know", "Somewhat difficult", "Somewhat easy","Very difficult","Very easy"), 
                                       labels = c(0,1,2,3,4))

# NA values detection and deleting the row.
sapply(data, function(x) sum(is.na(x)))
data <- data[!is.na(data$work_interfere),]
data

#selecting only data avaiable in united states
data<-subset(data, Country=="United States")
data

#removing non required fields
survey.df<-data[-1][-2][-5]

#Logistic Regression

# partition data
set.seed(2)
train.index <- sample(c(1:dim(survey.df)[1]), dim(survey.df)[1]*0.6)
train.df <- survey.df[train.index, ]
valid.df <- survey.df[-train.index, ]


logit.reg <- glm(train.df$treatment ~ ., data = train.df, family = binomial(link='logit')) 
options(scipen=999) # remove scientific notation
summary(logit.reg)
#install.packages("jtools")
library(jtools)
summ(logit.reg, digits=5)

# Plot barchart of coefficients
library(lattice)
barchart(logit.reg$coefficients)

# use predict() with type = "response" to compute predicted probabilities.Response is used to predict probablities 
logit.reg.pred <- predict(logit.reg, valid.df, type = "response")

confusionMatrix(as.factor(ifelse(logit.reg.pred > 0.5, 1, 0)), 
                as.factor(valid.df$treatment))

anova(logit.reg, test="Chisq")
library(ROCR)
p <- predict(logit.reg, newdata=valid.df, type="response")
pr <- prediction(p, valid.df$treatment)
prf <- performance(pr, measure = "tpr", x.measure = "fpr")
plot(prf)

auc <- performance(pr, measure = "auc")
auc <- auc@y.values[[1]]
auc


# actual and predicted records
Plot.Pred<-data.frame(actual = valid.df$treatment, logit.pred = logit.reg.pred)
head(Plot.Pred,10)

# plot actual and predicted values
Plot.Pred<-Plot.Pred[order(Plot.Pred$logit.pred),]
Plot.Pred$x<-c(seq(1,243))
with(Plot.Pred, plot(x, actual, col="red3"))
par(new = T)
with(Plot.Pred, plot(x, logit.pred, col="black"))


# Outlier detection using Cook's Distance
cooksd <- cooks.distance(logit.reg)

plot(cooksd, pch="*", cex=2, main="Influential Obs by Cooks distance")  # plot cook's distance
abline(h = 4*mean(cooksd, na.rm=T), col="red")  # add cutoff line
text(x=1:length(cooksd)+1, y=cooksd, labels=ifelse(cooksd>4*mean(cooksd, na.rm=T),names(cooksd),""), col="red")  #
#-------------------------------------------------------------------------------------------------------------------------------------------#
## random forest


library(randomForest)
rf <- randomForest(as.factor(train.df$treatment) ~ ., data = train.df, ntree = 500, 
                   mtry = 4, nodesize = 5, importance = TRUE)  
summary(rf)
head(rf$votes,10)


min_depth_frame <- min_depth_distribution(rf)
save(min_depth_frame, file = "min_depth_frame.rda")
plot_min_depth_distribution(min_depth_frame)

## Plot forest by prediction errors
plot(rf)
legend("top", colnames(rf$err.rate),cex=0.8,fill=1:3)


## variable importance plot
varImpPlot(rf, type = 1)


## confusion matrix
rf.pred <- predict(rf, valid.df)
library(caret)
confusionMatrix(rf.pred, as.factor(valid.df$treatment))

#----------------------------------------------------------------------------------------------------------------------#

# run naive bayes
Survey.nb <- naiveBayes(train.df$treatment ~ ., data = train.df)
Survey.nb


## predict probabilities: Training
pred.prob <- predict(Survey.nb, newdata = train.df, type = "raw")
## predict class membership
pred.class <- predict(Survey.nb, newdata = valid.df)
confusionMatrix(pred.class, valid.df$treatment)

plot(pred.class)
#-------------------------------------------------------------------------------------------------------------------------#

#ensemble

# bagging
bag <- bagging(treatment ~ ., data = train.df)
pred <- predict(bag, valid.df, type = "class")
confusionMatrix(as.factor(pred$class), valid.df$treatment)


# boosting
boost <- boosting(treatment ~ ., data = train.df)
predb <- predict(boost, valid.df, type = "class")
confusionMatrix(as.factor(predb$class), valid.df$treatment)

#-------------------------------------------------------------------------------------------------------------------------#

#neural network
#changing predictors into numerics

attach(survey.df)
survey.df$Gender = as.numeric(survey.df$Gender)
survey.df$work_interfere = as.numeric(survey.df$work_interfere)
survey.df$family_history = as.numeric(survey.df$family_history)
survey.df$leave = as.numeric(survey.df$leave)
survey.df$supervisor = as.numeric(survey.df$supervisor)
survey.df$coworkers = as.numeric(survey.df$coworkers)
survey.df$mental_health_consequence = as.numeric(survey.df$mental_health_consequence)
survey.df$mental_health_interview = as.numeric(survey.df$mental_health_interview)
survey.df$mental_vs_physical = as.numeric(survey.df$mental_vs_physical)
survey.df$phys_health_consequence = as.numeric(survey.df$phys_health_consequence)
survey.df$phys_health_interview = as.numeric(survey.df$phys_health_interview)
survey.df$obs_consequence = as.numeric(survey.df$obs_consequence)
survey.df$treatment = as.numeric(survey.df$treatment)
survey.df$wellness_program = as.numeric(survey.df$wellness_program)
survey.df$tech_company = as.numeric(survey.df$tech_company)
survey.df$remote_work = as.numeric(survey.df$remote_work)
survey.df$anonymity = as.numeric(survey.df$anonymity)
survey.df$benefits = as.numeric(survey.df$benefits)
survey.df$care_options = as.numeric(survey.df$care_options)
survey.df$seek_help = as.numeric(survey.df$seek_help)
scaleddata<-scale(survey.df)

#to normalize predictors

normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}


maxmindf <- as.data.frame(lapply(survey.df, normalize))
# Training and Test Data
trainset <- maxmindf[1:363, ]
testset <- maxmindf[364:606, ]


#Neural Network

nn <- neuralnet(treatment ~ ., data=trainset, hidden=c(2,1), linear.output=FALSE, threshold=0.01)
nn$result.matrix
plot(nn)
temp_test <- subset(testset, select = c("Gender","work_interfere", "family_history", "leave", "supervisor","coworkers","mental_health_consequence","mental_health_interview","mental_vs_physical","phys_health_consequence","phys_health_interview","obs_consequence","wellness_program","tech_company","remote_work","anonymity","benefits","care_options","seek_help"))
head(temp_test)
nn.results <- neuralnet::compute(nn, temp_test)
results <- data.frame(actual = testset$treatment, prediction = nn.results$net.result)
results


roundedresults<-sapply(results,round,digits=0)
roundedresultsdf=data.frame(roundedresults)
attach(roundedresultsdf)
table(actual,prediction)
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
#knn

new.norm.values <- preProcess(trainset, method=c("center", "scale"))
new.norm.train.df <- predict(new.norm.values, newdata = trainset)
new.norm.valuesq <- preProcess(testset, method=c("center", "scale"))
new.norm.test.df <- predict(new.norm.values, newdata = testset)



#initialize a data frame with two columns: k, and accuracy with seq from 1-14 and rmse value 0 for 14 repeatations
RMSE.df <- data.frame(k = seq(1, 14, 1), RMSE = rep(0, 14))

# compute knn for different k on validation.
for(i in 1:14){
  new.knn.pred <- class::knn(train = new.norm.train.df,
                             test = new.norm.test.df,
                             cl = trainset$treatment, k = i)
  RMSE.df[i,2]<-RMSE(as.numeric(as.character(new.knn.pred)),trainset$treatment)
  
}

RMSE.df
#RMSE plot
plot(RMSE.df, type="b", xlab="K- Value",ylab="RMSE level")


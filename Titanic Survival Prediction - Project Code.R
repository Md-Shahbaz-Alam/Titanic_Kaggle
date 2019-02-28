# Titanic Survival Prediction

# Loading the required libraries
library(Hmisc)
library(randomForest)
library(caret)
library(gbm)

# Reading the Data set in the directory
titanic_train <- read.csv("titanic-train.csv", header = T, sep = ",")
titanic_test <- read.csv("titanic-test.csv", header = T, sep = ",")


# Checking the details of the column
str(titanic_data)
summary(titanic_data)


# Creating Survived column with NA values in test data set 
# as it is not present in the test dataset
titanic_test$Survived <- rep(NA,nrow(titanic_test))


# Creating a dummy column for keeping track of train and test data set in case if suffled
# Here 1 is assigned to data belongs to train and 2 is assigned to data belongs to test
titanic_train$dummy <- rep(1,nrow(titanic_train))
titanic_test$dummy <- rep(2,nrow(titanic_test))


# Combining the data set for preprocessing
titanic_data <- rbind(titanic_train,titanic_test)


# Changing the class of the column in the data set as required
titanic_data$Embarked <- as.factor(titanic_data$Embarked)
titanic_data$Pclass <- as.factor(titanic_data$Pclass)
titanic_data$Sex <- as.factor(titanic_data$Sex)
titanic_data$Survived <- as.factor(titanic_data$Survived)


# Creating new column in the dataset for the number of family
titanic_data$N_Family <- titanic_data$SibSp+titanic_data$Parch+1
titanic_data$individual_price <- titanic_data$Fare/titanic_data$N_Family


# Imputing NAs values 
titanic_data$individual_price[1044] <- mean(titanic_data$individual_price[
  which(titanic_data$Pclass == 3 & is.na(titanic_data$Fare)==FALSE)])
titanic_data$Embarked[which(titanic_data$Embarked=="")] <- "S"


# Subsetting the title from the name of the data set
titanic_data$Title <- gsub('(.*, )|(\\..*)', '', titanic_data$Name)
titanic_data$Title[titanic_data$Title == 'Ms'] <- 'Miss'


# Imputing the missing values in the Age column
titanic_data$Age[titanic_data$Title == 'Dr' & is.na(titanic_data$Age) ==TRUE] <- 
  mean(titanic_data$Age[titanic_data$Title == 'Dr' & is.na(titanic_data$Age)==FALSE])

titanic_data$Age[titanic_data$Title == 'Master'& is.na(titanic_data$Age) ==TRUE] <- 
  mean(titanic_data$Age[titanic_data$Title == 'Master' & is.na(titanic_data$Age)==FALSE])

titanic_data$Age[titanic_data$Title == 'Miss'& is.na(titanic_data$Age)==TRUE] <- 
  mean(titanic_data$Age[titanic_data$Title == 'Miss' & is.na(titanic_data$Age)==FALSE])

titanic_data$Age[titanic_data$Title == 'Mr'& is.na(titanic_data$Age)==TRUE] <- 
  mean(titanic_data$Age[titanic_data$Title == 'Mr' & is.na(titanic_data$Age)==FALSE])

titanic_data$Age[titanic_data$Title == 'Mrs'& is.na(titanic_data$Age)==TRUE] <- 
  mean(titanic_data$Age[titanic_data$Title == 'Mrs' & is.na(titanic_data$Age)==FALSE])

titanic_data$Fare <- with(titanic_data, impute(Fare, 14.454))

# Checking for correlation of the column with the target column
chisq.test(titanic_data$Survived, titanic_data$Pclass)
chisq.test(titanic_data$Survived, titanic_data$Sex)
chisq.test(titanic_data$Survived, titanic_data$Cabin)
chisq.test(titanic_data$Survived, titanic_data$Embarked)
chisq.test(titanic_data$Survived, titanic_data$Ticket)

a1 <- aov(Age~Survived, data = titanic_data)
summary(a1)
a2 <- aov(N_Family~Survived, data = titanic_data)
summary(a2)
a3 <- aov(individual_price~Survived, data = titanic_data)
summary(a3)

# Dividing the dataset into train and test
titanic_data_train <- titanic_data[which(titanic_data$dummy==1),]
titanic_data_test <- titanic_data[which(titanic_data$dummy==2),]


# Removing the dummy column which was created earlier for reference
titanic_data_train <- within(titanic_data_train, rm(dummy))
titanic_data_test <- within(titanic_data_test, rm(dummy))


# Creating model using glm method
fit_glm <- glm(Survived ~ Pclass+Sex+Age+N_Family,
               family= binomial (link="logit"), 
               data = titanic_data_train)

summary(fit_glm)


# Creating model using random forest modeling
fit_rf <- randomForest(Survived ~ Pclass+Sex+Age+N_Family,
                       ntree=200,
                       n_estimator = 100,
                       min_samples_split=4,
                       oob_score = T,
                       titanic_data_train)
fit_rf


# Creating gbm model using caret library
fit_gbm <- train(Survived ~ Pclass+Sex+Age+N_Family,
               data=titanic_data_train,
               metric='Accuracy',
               method='gbm',
               verbose=TRUE)
fit_gbm


# Predicting target using Random Forest in the train Data set
# As the error in the model is low compare to other models
pred_rf  <- predict(fit_rf, titanic_data_train)
pred_rf  <- as.factor(pred_rf)
confusionMatrix(pred_rf, titanic_data_train$Survived)


# Predicting on test Data Set
pred_rf_test <- predict(fit_rf, titanic_data_test)
pred_rf_test  <- as.factor(pred_rf_test)


# Creating a Data Frame with the PassengerIDs of passenger in test data set and
# assigning the survival predicted value corresponding to that IDs
output <- data.frame(PassengerID = titanic_data_test$PassengerId, Survived = pred_rf_test)


# Writing the output result into file
write.csv(output, "Titanic-Survival_prediction.csv")


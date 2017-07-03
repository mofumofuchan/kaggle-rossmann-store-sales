# Load Dependeicies
library(caret)
library(randomForest)
library(readr)
library(lubridate)
library(plyr)
library(doMC)
registerDoMC(cores = 6)

# Set seed
set.seed(1337)

train <- read_csv("../input/train.csv",
                  col_names=TRUE, col_types = cols("i", "i", 
                                         "D", "i",
                                         "i", "i", "i",
                                         "c", "i"))
test  <- read_csv("../input/test.csv",
                  col_names=TRUE, col_types = cols("i", "i", 
                                                   "i", "D",
                                                   "i", "i", 
                                                   "c", "i"))
store <- read_csv("../input/store.csv")

# Merge Store stuff
train <- merge(train, store, by=c('Store'))
test <- merge(test, store, by=c('Store'))
test <- arrange(test,Id)

# Only evaludated on Open days
train <- train[ which(train$Open=='1'),]
train$Open <- NULL
test$Open <- NULL

# Set missing data to 0
train[is.na(train)] <- 0
test[is.na(test)] <- 0

# Date stuff
train$Date <- ymd(train$Date)
test$Date <- ymd(test$Date)

train$month <- as.factor(month(train$Date))
test$month <- as.factor(month(test$Date))

train$year <- as.factor(year(train$Date))
test$year <- as.factor(year(test$Date))

#Factorize stuff
train$DayOfWeek <- as.factor(train$DayOfWeek)
test$DayOfWeek <- as.factor(test$DayOfWeek)

#train$Open <- as.factor(train$Open)
#test$Open <- as.factor(test$Open)

train$Promo <- as.factor(train$Promo)
test$Promo <- as.factor(test$Promo)

train$SchoolHoliday <- as.factor(train$SchoolHoliday)
test$SchoolHoliday <- as.factor(test$SchoolHoliday)

# always 1 for training and test
train$StateHoliday <- NULL
test$StateHoliday <- NULL

# Factorize store stuff
train$StoreType <- as.factor(train$StoreType)
test$StoreType <- as.factor(test$StoreType)

train$Assortment <- as.factor(train$Assortment)
test$Assortment <- as.factor(test$Assortment)

train$CompetitionDistance <- as.numeric(train$CompetitionDistance)
test$CompetitionDistance <- as.numeric(test$CompetitionDistance)


# target variables
train$Sales <- as.numeric(train$Sales)
train$Customers <- NULL #as.numeric(train$Customers)

train$Sales <- log(train$Sales+1)


fitControl <- trainControl(method="cv", number=3, verboseIter=T)
rfFit <- train(Sales ~., 
               method="rf", data=train, ntree=50, importance=TRUE,
               sampsize=100000,
               do.trace=10, trControl=fitControl)


pred <- predict(rfFit, test)
submit = data.frame(Id = test$Id, Sales = (exp(pred)) -1)

write.csv(submit, "../submissions/rf_starter_with_log_exp.csv", row.names = FALSE, quote = FALSE)

library(xgboost)
library(lubridate)
library(plyr)

set.seed(1337)

cat("reading the train and test data\n")
train <- read.csv("../input/train.csv", stringsAsFactors = F)
test  <- read.csv("../input/test.csv", stringsAsFactors = F)
store <- read.csv("../input/store.csv", stringsAsFactors = F)

# Merge Store data
train <- merge(train,store)
test <- merge(test,store)

test$Open[is.na(test$Open)] <- 1
train <- train[ which(train$Open=='1'),]
train <- train[ which(train$Sales!='0'),]

# Date stuff
train$Date <- ymd(train$Date)
test$Date <- ymd(test$Date)
train$day <- as.integer(day(train$Date))
test$day <- as.integer(day(test$Date))
train$week <- as.integer(week(train$Date))
test$week <- as.integer(week(test$Date))
train$month <- as.integer(month(train$Date))
test$month <- as.integer(month(test$Date))
train$year <- as.integer(year(train$Date))
test$year <- as.integer(year(test$Date))
train$doy <- as.integer(yday(train$Date))
test$doy <- as.integer(yday(test$Date))
train$Date <- NULL
test$Date <- NULL

#Factorize stuff
#train$DayOfWeek <- as.factor(train$DayOfWeek)
#test$DayOfWeek <- as.factor(test$DayOfWeek)
#train$Promo <- as.factor(train$Promo)
#test$Promo <- as.factor(test$Promo)
#train$SchoolHoliday <- as.factor(train$SchoolHoliday)
#test$SchoolHoliday <- as.factor(test$SchoolHoliday)
#train$Open <- as.factor(train$Open)
#test$Open <- as.factor(test$Open)
#train$Promo2 <- as.factor(train$Promo2)
#test$Promo2 <- as.factor(test$Promo2)

train$CompetitionOpenSinceMonth[is.na(train$CompetitionOpenSinceMonth)] <- 8
test$CompetitionOpenSinceMonth[is.na(test$CompetitionOpenSinceMonth)] <- 8
train$CompetitionOpenSinceYear[is.na(train$CompetitionOpenSinceYear)] <- 2009
test$CompetitionOpenSinceYear[is.na(test$CompetitionOpenSinceYear)] <- 2009

competition_start <- strptime('20.10.2015', format='%d.%m.%Y')
train$CompetitionDaysOpen <- as.numeric(
  difftime(competition_start,
           strptime(paste('1',
                          train$CompetitionOpenSinceMonth,
                          train$CompetitionOpenSinceYear, sep = '.'),
                    format='%d.%m.%Y'), units='days'))
test$CompetitionDaysOpen <- as.numeric(
  difftime(competition_start,
           strptime(paste('1',
                          test$CompetitionOpenSinceMonth,
                          test$CompetitionOpenSinceYear, sep = '.'),
                    format='%d.%m.%Y'), units='days'))
train$CompetitionDaysOpen[is.na(train$CompetitionDaysOpen)] <- mean(train$CompetitionDaysOpen, na.rm = T)
test$CompetitionDaysOpen[is.na(test$CompetitionDaysOpen)] <- mean(test$CompetitionDaysOpen, na.rm = T)

train$CompetitionDistance[is.na(train$CompetitionDistance)] <- mean(train$CompetitionDistance, na.rm = T)
test$CompetitionDistance[is.na(test$CompetitionDistance)] <- mean(test$CompetitionDistance, na.rm = T)

train$Promo2Days <- as.numeric(
  difftime(competition_start,
           strptime(paste('1',
                  train$Promo2SinceWeek,
                  train$Promo2SinceYear,
                  sep = '.'),
            format='%d.%m.%Y'), units='days'))
test$Promo2Days <- as.numeric(
  difftime(competition_start,
           strptime(paste('1',
                  test$Promo2SinceWeek,
                  test$Promo2SinceYear, sep = '.'),
            format='%d.%m.%Y'), units='days'))
train$Promo2Days[is.na(train$Promo2Days)] <- 0
test$Promo2Days[is.na(test$Promo2Days)] <- 0

train$Promo2SinceWeek <- NULL
train$Promo2SinceYear <- NULL
test$Promo2SinceWeek <- NULL
test$Promo2SinceYear <- NULL

sapply(train, function(x) length(which(is.na(x))))
sapply(test, function(x) length(which(is.na(x))))


# Google trends
trends_files_path = '../input/trends/'
trends_files <- list.files(trends_files_path)

for(trends_file in trends_files) {
  print(trends_file)
  trends <- read.csv(paste(trends_files_path, trends_file, sep = ''))
  names(trends)[2] <- gsub('.csv', '', gsub('google_trends_', '', trends_file))
  if (!exists('all.trends')){
    all.trends <- trends
  }
  else {
    print('merging')
    all.trends <- merge(all.trends, trends, by = c('Week'))
  }
}

all.trends$week <- week(all.trends$Week)
all.trends$year <- year(all.trends$Week)
all.trends$Week <- NULL
train <- merge(train, all.trends, by=c("week", "year"))
test <- merge(test, all.trends, by=c("week", "year"))

feature.names <- names(train)[c(1:4,7:38)]
cat("Feature Names\n")
feature.names

cat("train data column names after slight feature engineering\n")
names(train)
for (f in feature.names) {
  if (class(train[[f]])=="character") {
    levels <- unique(c(train[[f]], test[[f]]))
    train[[f]] <- as.integer(factor(train[[f]], levels=levels))
    test[[f]]  <- as.integer(factor(test[[f]],  levels=levels))
  }
}

cat("test data column names after slight feature engineering\n")
names(test)
tra<-train[,feature.names]
nrow(train)
h<-sample(nrow(train),50000)

RMPSE<- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  elab<-exp(as.numeric(labels))-1
  epreds<-exp(as.numeric(preds))-1
  err <- sqrt(mean((epreds/elab-1)^2))
  return(list(metric = "RMPSE", value = err))
}

dval<-xgb.DMatrix(data=data.matrix(tra[h,]),label=log(train$Sales+1)[h])
dtrain<-xgb.DMatrix(data=data.matrix(tra[-h,]),label=log(train$Sales+1)[-h])
watchlist<-list(val=dval,train=dtrain)
param <- list(  objective           = "reg:linear",
                booster = "gbtree",
                eta                 = 0.02,
                max_depth           = 10,
                subsample           = 0.9,
                colsample_bytree    = 0.7
                #num_parallel_tree   = 2
                # alpha = 0.0001,
                # lambda = 1
)

clf <- xgb.train(   params              = param,
                    data                = dtrain,
                    nrounds             = 3000,
                    verbose             = 0,
                    early.stop.round    = 100,
                    watchlist           = watchlist,
                    maximize            = FALSE,
                    feval=RMPSE
)

importance_matrix <- xgb.importance(feature.names, model = clf)
xgb.plot.importance(importance_matrix)

pred1 <- exp(predict(clf, data.matrix(test[,feature.names]))) -1
submission <- data.frame(Id=test$Id, Sales=pred1)
cat("saving the submission file\n")
write_csv(submission, "../submissions/xgboost_with_google_trends.csv")

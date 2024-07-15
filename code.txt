library(vip)
library(RColorBrewer)
library(ggplot2)
library(data.table)
library(class)
library(caret)
library(rpart) 
library(randomForest) 
library(e1071) 
library(caTools) 
library(readxl)
library(rstudioapi)
library(gridExtra)
library(corrplot)

dir <- dirname(rstudioapi::getActiveDocumentContext()$path)
setwd(dir)

dataset <- read.csv("classification project as csv.csv", header = T)

# Checking the data types
str(dataset)

# Dates are not in the appropriate format so we will convert them. 
# We will then separate them into three columns (day, month, year)
dataset$date.of.reservation <- as.Date(dataset$date.of.reservation, format = "%m/%d/%Y")
dataset$day <- as.numeric(format(dataset$date.of.reservation, "%d"))
dataset$year <- as.numeric(format(dataset$date.of.reservation, "%Y"))
dataset$month <- as.numeric(format(dataset$date.of.reservation, "%m"))

# Two dates are in weird format, we'll just ignore them since they are just 2 observations.
# There are no other NA values in the dataset.
print(colSums(is.na(dataset)))
dataset <- na.omit(dataset)

# Outliers detection
numeric.only <- sapply(dataset, is.numeric)
y <- dataset[, numeric.only, drop = FALSE]
par(mfrow=c(1,1))
boxplot(y)

for (i in 1:length(y)){
    y1<-y[,i]
    out <- boxplot( y1, plot=FALSE )$out
    if(length(out)!=0){
        print('-------------------------------------------------------')
        print( paste('Outliers for variable', names(y)[i] ) )
        print( paste(length(out), 'outliers') )
        print( paste(round(100*length(out)/sum(!is.na(y1)),1),
                     '% outliers', sep='' ) )
        print(which( y1 %in% out ))
    }
}
# Finding the number of unique elements per column. 
# This helps to detect categorical values. We already have a good idea 
# on them but we are double checking. 
print(sapply(dataset, function(x) length(unique(x))))

# Binary are: repeated, booking.status 

# Choosing columns to drop. ID should be of no help.
# We will also drop the date column since we transformed it.
dropped_columns <- c("Booking_ID", "date.of.reservation")
# Dropping columns
dataset = dataset[! colnames(dataset) %in% dropped_columns]
# Verifying columns were dropped
str(dataset)

##############################################
##
## Histograms for the quantitative variables
##
##############################################

my.hist <- function(dataset, col.name, b=30) {
    ggplot(dataset, aes(x= !!sym(col.name))) + 
        geom_histogram(bins=b, fill='lightblue', color='black')
}

hist1 <- my.hist(dataset, "number.of.adults", 4)
hist2 <- my.hist(dataset, "number.of.children",4)
hist3 <- my.hist(dataset, "average.price")
hist4 <- my.hist(dataset, "month",12) + scale_x_continuous(breaks = 1:12, labels = c("1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"))

grid.arrange(hist1, hist2, hist3, hist4, ncol = 2)

##############################################
##
## Donut plots for Groups
##
##############################################

my_donut_plot <- function(data, group, title, palette = c("#08519c","#3182bd","#6baed6","#9ecae1","#c6dbef","#eff3ff")) {
    group_count = length(unique(data[[group]]))
    
    data_with_proportions <- data[, .(proportion = round(.N / nrow(data), digits = 4)), keyby = group]
    ggplot(data_with_proportions, aes(x = 2, y = proportion, fill = !!sym(group))) +
        geom_bar(stat = "identity", color = "white") +
        geom_text(aes(label = paste(proportion*100, "%")), position = position_stack(vjust=0.5), size=4, color = "black")+
        coord_polar("y", start = 0) +
        theme_void() +
        scale_fill_manual(values = palette[1:group_count]) +
        labs(title = title) +
        xlim(0.5, 2.5) + 
        theme(legend.text = element_text(size = 14))
}

put.other <- function(coldata, levels.to.keep=5) {
    levels <- names(sort(table(coldata), decreasing = T))[1:levels.to.keep]
    coldata[!coldata %in% levels]="Other"
    return(coldata)
}

copy.df <- data.frame(dataset)
copy.df$room.type <- put.other(copy.df$room.type, 2)
copy.df$market.segment.type <- put.other(copy.df$market.segment.type, 2)


donut1 <- my_donut_plot(data.table(copy.df), "type.of.meal", "Type of Meal")
donut2 <- my_donut_plot(data.table(copy.df), "room.type", "Type of room")
donut3 <- my_donut_plot(data.table(copy.df), "market.segment.type", "Type of market segment")
donut4 <- my_donut_plot(data.table(copy.df), "booking.status", "Status of the booking")

grid.arrange(donut1, donut2, ncol = 2)
grid.arrange(donut3, donut4, ncol = 2)

##############################################
##
# Pairwise Comparisons
##
##############################################

# Converting the target variable to binary 
dataset$booking.status <- ifelse(dataset$booking.status == 'Canceled', '1', '0')

# Compute correlation coefficients
copy.df <- data.frame(dataset)
copy.df$booking.status <- as.numeric(as.factor(copy.df$booking.status))

cor_matrix <- cor(copy.df[, sapply(copy.df, is.numeric), drop = FALSE], use = "complete.obs")

# Sort the absolute correlation coefficients with respect to 'booking.status'
sorted_cor <- sort(abs(cor_matrix["booking.status",]), decreasing = TRUE)

# Create a dataframe for plotting
cor_df <- data.frame(
    feature = names(sorted_cor),
    correlation = cor_matrix["booking.status", names(sorted_cor)]
)

# Sort the dataframe by correlation in increasing order
cor_df <- cor_df[order(cor_df$correlation), ]

cor_df$feature <- factor(cor_df$feature, levels = cor_df$feature)

palette <- brewer.pal(n = 16, name = "Blues")

ggplot(cor_df, aes(x = 1, y = feature, fill = correlation)) +
    geom_tile() +
    scale_fill_gradientn(colors = palette) +
    geom_text(aes(label = round(correlation, 2)), vjust = 1) +
    theme_minimal() +
    theme(
        axis.text.x = element_blank(),
        axis.ticks.x = element_blank(),
        axis.title.x = element_blank(),
        axis.title.y = element_blank(),
        legend.position = "none"
    ) +
    labs(
        title = "Correlation Coefficient Between Booking Status and Numeric or Binary Feature",
        y = "Features"
    ) +
    coord_fixed(ratio = 0.1)


##############################################

# Making sure cyclic nature of months is expressed in our transformation. 

dataset$month_sin <- sin(2 * pi * dataset$month / 12)
dataset$month_cos <- cos(2 * pi * dataset$month / 12)
dataset <- subset(dataset, select = -month)
str(dataset)

# Defining the factor columns
factor.cols <- c("type.of.meal", "room.type", "market.segment.type",
                 "repeated")

# Converting factor columns
dataset[factor.cols] <- lapply(dataset[factor.cols], function(x) as.numeric(as.factor(x)))

# Verifying that column types have changed
str(dataset)


target_column <- "booking.status"
feature_cols <- colnames(dataset[colnames(dataset) != target_column])

# All feature cols have been transformed to numeric
numeric_cols_boolean = feature_cols

# Make target column a factor
dataset$booking.status = as.factor(dataset$booking.status)
str(dataset)
##############################################
##
## Splitting and Scaling
##
##############################################

set.seed(42) 
split = sample.split(dataset$booking.status, SplitRatio = 0.75) 
training_set = subset(dataset, split == TRUE) 
test_set = subset(dataset, split == FALSE)

mean_values <- apply(training_set[numeric_cols_boolean], 2, mean)
variance_values <- apply(training_set[numeric_cols_boolean], 2, var)

# scaling 
training_set[numeric_cols_boolean] = scale(training_set[numeric_cols_boolean]) 
test_set[numeric_cols_boolean] = scale(test_set[numeric_cols_boolean],
                                       center = mean_values, scale = sqrt(variance_values)) 

##############################################
## 
## With Cross Val and Grid Search
##
##############################################
f1 <- function(data, lev = NULL, model = NULL) {
    # Check if data contains necessary columns
    if (!all(c("pred", "obs") %in% names(data))) {
        stop("data must contain 'pred' and 'obs' columns")
    }
    
    # Get levels of 'obs' to construct confusion matrix
    obs_levels <- levels(data$obs)
    
    # Calculate confusion matrix
    conf_mat <- table(data$pred, data$obs)
    
    # Initialize variables for precision, recall, and F1 score
    precision <- numeric(length(obs_levels))
    recall <- numeric(length(obs_levels))
    f1_val <- numeric(length(obs_levels))
    
    # Calculate precision, recall, and F1 score for each level
    for (i in 1:length(obs_levels)) {
        tp <- as.numeric(conf_mat[i, i])
        fp <- sum(conf_mat[i, ]) - tp
        fn <- sum(conf_mat[, i]) - tp
        
        precision[i] <- ifelse(tp + fp == 0, 0, tp / (tp + fp))
        recall[i] <- ifelse(tp + fn == 0, 0, tp / (tp + fn))
        
        f1_val[i] <- ifelse(precision[i] + recall[i] == 0, 0, (2 * precision[i] * recall[i]) / (precision[i] + recall[i]))
    }
    
    # Return mean F1 score
    mean_f1 <- mean(f1_val, na.rm = TRUE)
    names(mean_f1) <- c("F1")
    
    return(mean_f1)
}

calculate_f1_confusion_matrix <- function(predictions, true_labels, positive_label) {
    # Calculate confusion matrix
    cm <- confusionMatrix(predictions, true_labels, positive = positive_label)
    
    # Extract true positives (TP), false positives (FP), and false negatives (FN)
    TP <- cm$table[positive_label, positive_label]
    FP <- sum(cm$table[positive_label, ]) - TP
    FN <- sum(cm$table[, positive_label]) - TP
    
    # Calculate precision, recall, and F1 score
    precision <- ifelse(TP + FP == 0, 0, TP / (TP + FP))
    recall <- ifelse(TP + FN == 0, 0, TP / (TP + FN))
    f1_score <- ifelse(precision + recall == 0, 0, (2 * precision * recall) / (precision + recall))
    
    # Add F1 score to confusion matrix
    cm$byClass["F1"] <- f1_score
    
    return(cm)
}

control <- trainControl(method = "repeatedcv", 
                        number = 10,
                        repeats = 5,
                        savePredictions = "final", 
                        classProbs = F,
                        verboseIter = TRUE,
                        summaryFunction = f1)
                       

##############################################
##
## KNN algorithm
##
##############################################

set.seed(42) 
model_knn <- train(booking.status ~ ., 
                   data = training_set, 
                   method = "knn",
                   metric = "F1",
                   trControl = control,
                   tuneGrid = expand.grid(k = 2:10))
model_knn$bestTune #3

predictions_knn <- predict(model_knn, test_set[feature_cols])

# Calculate confusion matrix
cm <- confusionMatrix(predictions_knn, test_set$booking.status, positive = "1")
cm <- calculate_f1_confusion_matrix(predictions_knn, test_set$booking.status, positive = "1")
cm
data.frame(Accuracy = cm$overall["Accuracy"],
           Sensitivity = cm$byClass["Sensitivity"],
           Specificity = cm$byClass["Specificity"],
           F1 = cm$byClass["F1"])


##############################################
##
## Plot for KNN (was not used in the report)
##
##############################################

set.seed(42) 
model_knn <- train(booking.status ~ ., 
                   data = training_set[, c("lead.time", "average.price", "booking.status")], 
                   method = "knn",
                   trControl = control,
                   metric = "F1",
                   tuneGrid = expand.grid(k = 2:10))

y_pred = predict(model_knn, newdata = test_set[, c("lead.time", "average.price", "booking.status")])
prob = predict(model_knn, newdata = test_set[, c("lead.time", "average.price", "booking.status")], type = "prob")
sum(test_set$booking.status == y_pred) / nrow(test_set)
print(table(test_set$booking.status, y_pred))

test <- test_set[, c("lead.time", "average.price", "booking.status")]

prob <- apply(prob, 1, max)
x_grid <- seq(min(test$lead.time), max(test$lead.time), length.out = 200)
y_grid <- seq(min(test$average.price), max(test$average.price), length.out = 200)
grid <- expand.grid(lead.time = x_grid, average.price = y_grid)

# Predict probabilities on the grid
prob_grid <- predict(model_knn, newdata = grid, type = "prob")
pred <- predict(model_knn, newdata = grid)

grid$y_pred = pred
grid$prob_0 = prob_grid[, "0"]
grid$prob_1 = prob_grid[, "1"]

# Create scatter plot
ggplot(grid, aes(x = lead.time, y = average.price, color = as.factor(y_pred))) +
    geom_point() +
    labs(title = "k-NN Predictions",
         x = "Lead Time",
         y = "Average Price",
         color = "Predicted Booking Status") +
    scale_color_manual(values = c("0" = "#f7c8c6", "1" = "#bae6f7", "green" = "blue", "orange" = "red")) +
    theme_minimal() + 
    geom_point(data = test, aes(x = lead.time, y = average.price, color = ifelse(booking.status == "1", "green", "orange")), size = 1)
    

test$lead.time

##############################################
##
# Decision Tree algorithm
##
##############################################
## Information Gain Model
##############################################

set.seed(42)
model_tree_information <- train(booking.status ~ ., 
                    data = training_set, 
                    method = "rpart",  
                    metric = "F1",
                    trControl = control,
                    parms = list(split = "information"),
                    tuneGrid = expand.grid(cp = seq(0.01, 0.5, 0.01)))
model_tree_information
predictions_tree_information = predict(model_tree_information, newdata = test_set[feature_cols]) 
cm1 <- calculate_f1_confusion_matrix(predictions_tree_information, test_set$booking.status, positive = "1")
cm1
data.frame(Accuracy = cm1$overall["Accuracy"],
           Sensitivity = cm1$byClass["Sensitivity"],
           Specificity = cm1$byClass["Specificity"],
           F1 = cm1$byClass["F1"])

# Create a variable importance plot
var_importance <- vip::vip(model_tree_information) # The 10 most important variables
print(var_importance)
varImp(model_tree_information)

##############################################
## Gini Impurity Model
##############################################

set.seed(42)
model_tree_gini <- train(booking.status ~ ., 
                    data = training_set, 
                    method = "rpart",  
                    trControl = control,
                    metric = "F1",
                    parms = list(split = "gini"),
                    tuneGrid = expand.grid(cp = seq(0.01, 0.5, 0.01)))

predictions_tree_gini = predict(model_tree_gini, newdata = test_set[feature_cols]) 
cm2 <- calculate_f1_confusion_matrix(predictions_tree_gini, test_set$booking.status, positive = "1")
cm2
data.frame(Accuracy = cm2$overall["Accuracy"],
           Sensitivity = cm2$byClass["Sensitivity"],
           Specificity = cm2$byClass["Specificity"],
           F1 = cm2$byClass["F1"])

# Create a variable importance plot
var_importance <- vip::vip(model_tree_gini)
print(var_importance)

#######################################################
##
## Random Forest algorithm
##
##############################################

set.seed(42)

model_rf <- train(booking.status ~ ., 
                  data = training_set, 
                  method = "rf",
                  metric = "F1",
                  trControl = control,
                  tuneGrid = expand.grid(mtry = 1:5))
model_rf

predictions_rf = predict(model_rf, newdata = test_set[feature_cols]) 
cm3 <- calculate_f1_confusion_matrix(predictions_rf, test_set$booking.status, positive = "1")
cm3
data.frame(Accuracy = cm3$overall["Accuracy"],
           Sensitivity = cm3$byClass["Sensitivity"],
           Specificity = cm3$byClass["Specificity"],
           F1 = cm3$byClass["F1"])


# Create a variable importance plot
var_importance <- vip::vip(model_rf)
print(var_importance)
plot(varImp(model_rf))

##############################################
##
## Feature importance
##
##############################################
#
# Use this in train function for faster training.
control1 <- trainControl(method = "cv", 
                         number = 10,
                         savePredictions = "final", 
                         classProbs = F,
                         verboseIter = TRUE,
                         summaryFunction = f1) 

results_list <- list()
for (i in 2:19) {
    top.columns <- c("booking.status",
                     "lead.time",
                     "average.price", 
                     "special.requests",
                     "day",
                     "month_cos",
                     "number.of.week.nights",
                     "market.segment.type",
                     "month_sin",
                     "number.of.weekend.nights",
                     "number.of.adults",
                     "year",
                     "type.of.meal",
                     "room.type",
                     "number.of.children",
                     "car.parking.space",
                     "repeated",
                     "P.not.C",
                     "P.C")[1:i]
    
    top.columns <- colnames(training_set) %in% top.columns
    
    # Retraining with fewer features
    set.seed(42)
    best_model_rf <- train(booking.status ~ .,
                      data = training_set[top.columns],
                      method = "rf",
                      metric = "F1",
                      trControl = control1, # Repeated CV was taking forever
                      tuneGrid = expand.grid(mtry = 1:5))

    predictions_best_rf = predict(best_model_rf, newdata = test_set[top.columns])
    confusion_matrix_best_rf <- calculate_f1_confusion_matrix(predictions_best_rf, test_set$booking.status, positive = "1")
    confusion_matrix_best_rf
    results_list[[i]] <- data.frame(Accuracy = confusion_matrix_best_rf$overall["Accuracy"],
               Sensitivity = confusion_matrix_best_rf$byClass["Sensitivity"],
               Specificity = confusion_matrix_best_rf$byClass["Specificity"],
               F1 = confusion_matrix_best_rf$byClass["F1"])

}
results_list 

##############################################
##
## Final Model
##
##############################################

set.seed(42)
final_rf <- train(booking.status ~ lead.time + average.price + special.requests + day +
                      month_cos + number.of.week.nights + market.segment.type, 
                  data = training_set, 
                  method = "rf",
                  metric = "F1",
                  trControl = control,
                  tuneGrid = expand.grid(mtry = 1:5))


predictions_final_rf = predict(final_rf, newdata = test_set[feature_cols]) 
cm_final <- calculate_f1_confusion_matrix(predictions_final_rf, test_set$booking.status, positive = "1")
cm_final
data.frame(Accuracy = cm_final$overall["Accuracy"],
           Sensitivity = cm_final$byClass["Sensitivity"],
           Specificity = cm_final$byClass["Specificity"],
           F1 = cm_final$byClass["F1"])


##############################################
##
## SVM
##
##############################################
## Linear SVM
##############################################

set.seed(42)
model_svm_linear <- train(booking.status ~ ., 
                   data = training_set, 
                   method = "svmLinear",
                   trControl = control,
                   metric = "F1",
                   # tuneGrid = expand.grid(C = seq(0.01, 2, 0.1)))
                    tuneGrid = expand.grid(C = c(0.001, 0.01, 0.1, 1, 10, 100)))

predictions_svm_linear <- predict(model_svm_linear, newdata = test_set[feature_cols])
cm4 <- calculate_f1_confusion_matrix(predictions_svm_linear, test_set$booking.status, positive = "1")
data.frame(Accuracy = cm4$overall["Accuracy"],
           Sensitivity = cm4$byClass["Sensitivity"],
           Specificity = cm4$byClass["Specificity"],
           F1 = cm4$byClass["F1"])

##############################################
## Non-Linear SVM
##############################################

set.seed(42)
model_svm <- train(booking.status ~ ., 
                   data = training_set, 
                   method = "svmRadial",
                   trControl = control, 
                   metric = "F1",
                   tuneGrid = expand.grid(C = c(0.01, 0.1, 1, 10),
                                           sigma = c(0.01,0.1, 1, 10, 100)))

predictions_svm <- predict(model_svm, newdata = test_set[feature_cols])
cm5 <- calculate_f1_confusion_matrix(predictions_svm, test_set$booking.status, positive = "1")
data.frame(Accuracy = cm5$overall["Accuracy"],
          Sensitivity = cm5$byClass["Sensitivity"],
          Specificity = cm5$byClass["Specificity"],
          F1 = cm5$byClass["F1"])

##############################################
## Weighted SVM, not used in the report
##############################################

set.seed(42)
w_model_svm <- train(booking.status ~ ., 
                   data = training_set, 
                   method = "svmRadialWeights",
                   trControl=control1,
                   tuneGrid = expand.grid(C = c(10),
                                          sigma = c(0.1),
                                          Weight = c(0.34)))

predictions_w_svm <- predict(w_model_svm, newdata = test_set[feature_cols])
cm6 <- calculate_f1_confusion_matrix(predictions_w_svm, test_set$booking.status, positive = "1")
data.frame(Accuracy = cm6$overall["Accuracy"],
           Sensitivity = cm6$byClass["Sensitivity"],
           Specificity = cm6$byClass["Specificity"],
           F1 = cm6$byClass["F1"])

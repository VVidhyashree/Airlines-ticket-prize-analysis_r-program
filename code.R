# Load the CSV file from the specified path
dataset <- read.csv("C:/Users/DELL/OneDrive/Desktop/Indian Airlines.csv")

# View the first few rows of the dataset
head(dataset)
column_names <- colnames(dataset)
print(column_names)

str(dataset)

shape <- dim(dataset)

# Print the shape
shape

# Get number of rows
num_rows <- nrow(dataset)

# Get number of columns
num_columns <- ncol(dataset)

# Print rows and columns
num_rows
num_columns


# Count the number of missing values in each column
missing_values_count <- sapply(dataset, function(x) sum(is.na(x)))

# Print the result
missing_values_count

# Load required libraries
library(ggplot2)

# Create the plot
ggplot(dataset, aes(x = airline, fill = airline)) +
  geom_bar() +
  labs(title = "Number of Flights by Airlines", y = "No. of Flights") +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 14),
    axis.title.x = element_blank()
  ) +
  theme(legend.position = "none") + # Remove the hue-like feature as it's redundant
  theme(plot.title = element_text(hjust = 0.5)) # Center the title
-------------------------------------------------------------------
  
  # Install and load ggplot2 if not already installed
install.packages("ggplot2")
library(ggplot2)

# Create the bar plot for 'class'
ggplot(dataset, aes(x = class, fill = class)) +
  geom_bar() +  # This creates the count plot
  labs(title = "Availability of Tickets w.r.t Class", y = "No. of Tickets") +  # Title and y-axis label
  theme_minimal() +  # Simple clean background
  theme(
    plot.title = element_text(hjust = 0.5, size = 14)  # Centering and sizing the title
  )
------------------------------------------------------------------------
  # Install and load necessary packages
install.packages("ggplot2")
library(ggplot2)
install.packages("dplyr")
library(dplyr)  # For data manipulation like sorting

# Sort the data by price in ascending order
sorted_data <- dataset %>% arrange(price)

# Create the bar plot
ggplot(sorted_data, aes(x = airline, y = price, fill = class)) +
  geom_bar(stat = "identity", position = "dodge") +  # Bar plot, separating by class
  labs(title = "Price for Different Airlines Based on Class", 
       x = "Airline", 
       y = "Price", 
       fill = "Class") +  # Title and axis labels
  theme_minimal() +  # Clean background
  theme(plot.title = element_text(hjust = 0.5, size = 14))  # Center and size the title

 
--------------------------------------------------------------------------------------------
df_ticket <- dataset %>%
group_by(duration) %>%
summarise(price = mean(price, na.rm = TRUE))

# Create the scatter plot
ggplot(df_ticket, aes(x = duration, y = price)) +
  geom_point(size = 3, color = 'blue') +  # Scatter plot points
  labs(title = "Average Price Depending on Duration of Flights", 
       x = "Duration", 
       y = "Price") +  # Title and axis labels
  theme_minimal() +  # Clean background
  theme(plot.title = element_text(hjust = 0.5, size = 14))  # Center the title
 
-----------------------------------------------------------------------------------

  
  # Bar plot for average duration based on stops
  ggplot(stop_duration, aes(x = factor(stops), y = average_duration, fill = factor(stops))) +
  geom_bar(stat = "identity") +
  labs(title = "Average Flight Duration Based on Number of Stops", x = "Number of Stops", y = "Average Duration (hrs)") +
  theme_minimal()
------------------------------------------------------------------------------------
  # Average price for airlines and stops
  airline_stop_stats <- dataset %>%
  group_by(airline, stops) %>%
  summarise(average_price = mean(price, na.rm = TRUE))

# Heatmap
ggplot(airline_stop_stats, aes(x = airline, y = stops, fill = average_price)) +
  geom_tile() +
  labs(title = "Heatmap of Average Price by Airlines and Stops", x = "Airline", y = "Number of Stops") +
  scale_fill_gradient(low = "lightblue", high = "darkblue") +
  theme_minimal()
---------------------------------------------------------------------------------------------
#Classification task 1

# Load required libraries
install.packages("rpart")
install.packages("randomForest")
install.packages("caTools")
library(rpart)
library(randomForest)
library(caTools)

# Preprocess dataset (assuming it's already loaded)
dataset$class <- as.factor(dataset$class)

# Split data into training and testing sets
set.seed(123)
split <- sample.split(dataset$class, SplitRatio = 0.7)
train_data <- subset(dataset, split == TRUE)
test_data <- subset(dataset, split == FALSE)

# 1. Decision Tree Model
tree_model <- rpart(class ~ airline + duration + stops + price, data = train_data, method = "class")
tree_pred <- predict(tree_model, test_data, type = "class")
tree_accuracy <- mean(tree_pred == test_data$class)

# 2. Random Forest Model
rf_model <- randomForest(class ~ airline + duration + stops + price, data = train_data, ntree = 100)
rf_pred <- predict(rf_model, test_data)
rf_accuracy <- mean(rf_pred == test_data$class)

# Print accuracies
print(paste("Decision Tree Accuracy:", round(tree_accuracy, 2)))
print(paste("Random Forest Accuracy:", round(rf_accuracy, 2)))

# View the first few rows of test_data to choose an example row
head(test_data)
#sample_input <- test_data[50, ]
sample_input <- test_data[nrow(test_data), ]

# Make a prediction using the Random Forest model
predicted_class_1 <- predict(rf_model, sample_input)
predicted_class_2 <- predict(tree_model, sample_input)

# Print the predicted class
print(paste("Predicted Ticket Class for Test Row 2:", predicted_class_1))
print(paste("Predicted Ticket Class for Test Row 2:", predicted_class_2))

# Compare with the actual class label for random forest model
actual_class <- test_data$class[nrow(test_data)]
print(paste("Actual Ticket Class for Test Row 2:", actual_class))
print(paste("Predicted Ticket Class for Test Row 2:", predicted_class_1))

# Compare with the actual class label for decision tree model
actual_class <- test_data$class[nrow(test_data)]
print(paste("Actual Ticket Class for Test Row 2:", actual_class))
print(paste("Predicted Ticket Class for Test Row 2:", predicted_class_1))


  
  # Convert stops to a factor for classification
  dataset$stops <- as.factor(dataset$stops)

# Split the data into training and testing sets
set.seed(456)
split_stops <- sample.split(dataset$stops, SplitRatio = 0.7)
train_data_stops <- subset(dataset, split_stops == TRUE)
test_data_stops <- subset(dataset, split_stops == FALSE)

# 1. Decision Tree for Stops Prediction
tree_model_stops <- rpart(stops ~ airline + duration + price + class, data = train_data_stops, method = "class")
tree_pred_stops <- predict(tree_model_stops, test_data_stops, type = "class")
tree_accuracy_stops <- mean(tree_pred_stops == test_data_stops$stops)

# 2. Random Forest for Stops Prediction
rf_model_stops <- randomForest(stops ~ airline + duration + price + class, data = train_data_stops, ntree = 200)
rf_pred_stops <- predict(rf_model_stops, test_data_stops)
rf_accuracy_stops <- mean(rf_pred_stops == test_data_stops$stops)

# Print the accuracies
print(paste("Decision Tree Accuracy for Stops Prediction:", round(tree_accuracy_stops, 2)))
print(paste("Random Forest Accuracy for Stops Prediction:", round(rf_accuracy_stops, 2)))

# Select a sample row from the test dataset
example_row <- test_data_stops[nrow(test_data_stops), ]  # Last row in test data

# Predict the number of stops using Decision Tree
tree_pred_example <- predict(tree_model_stops, example_row, type = "class")

# Predict the number of stops using Random Forest
rf_pred_example <- predict(rf_model_stops, example_row)

# Print the predictions
print(paste("Decision Tree Prediction for Stops:", tree_pred_example))
print(paste("Random Forest Prediction for Stops:", rf_pred_example))

# Compare with the actual value
actual_stops <- example_row$stops
print(paste("Actual Number of Stops:", actual_stops))

# Plot Feature Importance for Random Forest
importance <- importance(rf_model_stops)
varImpPlot(rf_model_stops, main = "Feature Importance for Stops Prediction")

# Convert 'stops' to integer labels starting from 0
train_data$stops <- as.numeric(factor(train_data$stops)) - 1
test_data$stops <- as.numeric(factor(test_data$stops)) - 1

# Update the labels
train_labels <- train_data$stops
test_labels <- test_data$stops

# Prepare the training and testing matrices
train_matrix <- as.matrix(train_data[, c("airline", "price", "duration", "class")])
test_matrix <- as.matrix(test_data[, c("airline", "price", "duration", "class")])

# Train the Gradient Boosting model
gbm_model <- xgboost(
  data = train_matrix,
  label = train_labels,
  nrounds = 100,  # Number of boosting rounds
  objective = "multi:softmax",  # Multiclass classification
  num_class = length(unique(train_labels)),  # Number of classes
  verbose = 0  # Suppress training logs
)

# Make predictions
gbm_predictions <- predict(gbm_model, test_matrix)

# Evaluate accuracy
gbm_accuracy <- mean(gbm_predictions == test_labels)
print(paste("Gradient Boosting Accuracy:", round(gbm_accuracy, 2)))


  # Ensure 'airline' is a factor and convert to integer labels starting from 0
  train_data$airline <- as.numeric(factor(train_data$airline)) - 1
test_data$airline <- as.numeric(factor(test_data$airline)) - 1

# Save the original airline mapping for interpretation later
airline_mapping <- levels(factor(dataset$airline))

# Define the target (airline) and features for training
train_labels <- train_data$airline
test_labels <- test_data$airline

train_matrix <- as.matrix(train_data[, c("price", "duration", "stops", "class")])
test_matrix <- as.matrix(test_data[, c("price", "duration", "stops", "class")])

install.packages("xgboost")
library(xgboost)
# Train the Gradient Boosting Model
gbm_model_airline <- xgboost(
  data = train_matrix,
  label = train_labels,
  nrounds = 100,  # Number of boosting rounds
  objective = "multi:softmax",  # Multiclass classification
  num_class = length(unique(train_labels)),  # Number of classes
  verbose = 0  # Suppress training logs
)

# Make predictions for test data
predicted_airlines <- predict(gbm_model_airline, test_matrix)

# Evaluate accuracy
airline_accuracy <- mean(predicted_airlines == test_labels)
print(paste("Gradient Boosting Accuracy for Airline Prediction:", round(airline_accuracy, 2)))


  # Load necessary libraries
  library(caret)
library(e1071)

# Load the dataset
flight_data <- read.csv("C:/Users/DELL/OneDrive/Desktop/Indian Airlines.csv")

# Select relevant columns
flight_data <- flight_data[, c("airline", "source_city", "departure_time", "stops", "duration", "days_left", "class")]

# Encode target variable (Economy = 0, Business = 1) and convert to factor
flight_data$class <- as.factor(ifelse(flight_data$class == "Economy", "Economy", "Business"))

# Convert categorical variables to factors
categorical_vars <- c("airline", "source_city", "departure_time", "stops")
flight_data[categorical_vars] <- lapply(flight_data[categorical_vars], as.factor)

# Split the data into training and testing sets
set.seed(42)
train_index <- createDataPartition(flight_data$class, p = 0.8, list = FALSE)
train_data <- flight_data[train_index, ]
test_data <- flight_data[-train_index, ]

# Train logistic regression model
logistic_model <- train(
  class ~ ., 
  data = train_data, 
  method = "glm", 
  family = "binomial",
  trControl = trainControl(method = "none")
)

# Make predictions on the test set
predictions <- predict(logistic_model, test_data)

# Convert predictions to factors with same levels as test_data$class
predictions <- factor(predictions, levels = levels(test_data$class))

# Evaluate the model
conf_matrix <- confusionMatrix(predictions, test_data$class)
accuracy <- conf_matrix$overall["Accuracy"]

# Print results
print(paste("Accuracy:", accuracy))
print(conf_matrix)






#Regression Tasks..
  # Load necessary libraries
  library(caret)
library(e1071)

# Load the dataset
flight_data <- read.csv("C:/Users/DELL/OneDrive/Desktop/Indian Airlines.csv")

# Select relevant columns for regression
regression_data <- flight_data[, c("airline", "source_city", "departure_time", "stops", "duration", "days_left", "price")]

# Convert categorical variables to factors
categorical_vars <- c("airline", "source_city", "departure_time", "stops")
regression_data[categorical_vars] <- lapply(regression_data[categorical_vars], as.factor)

# Handle missing values (if any)
regression_data <- na.omit(regression_data)

# Regression Task 1: Predict Flight Duration
set.seed(42)
train_index_1 <- createDataPartition(regression_data$duration, p = 0.8, list = FALSE)
train_data_1 <- regression_data[train_index_1, ]
test_data_1 <- regression_data[-train_index_1, ]

# Train a Gradient Boosting Model for Flight Duration
gbm_model_1 <- train(
  duration ~ airline + source_city + departure_time + stops + days_left + price,
  data = train_data_1,
  method = "gbm",
  trControl = trainControl(method = "cv", number = 5),
  verbose = FALSE
)

# Predict and evaluate
predictions_1 <- predict(gbm_model_1, test_data_1)
mse_1 <- mean((predictions_1 - test_data_1$duration)^2)
r2_1 <- cor(predictions_1, test_data_1$duration)^2

# Regression Task 2: Predict Ticket Price
set.seed(42)
train_index_2 <- createDataPartition(regression_data$price, p = 0.8, list = FALSE)
train_data_2 <- regression_data[train_index_2, ]
test_data_2 <- regression_data[-train_index_2, ]

# Train a Gradient Boosting Model for Ticket Price
gbm_model_2 <- train(
  price ~ airline + source_city + departure_time + stops + duration + days_left,
  data = train_data_2,
  method = "gbm",
  trControl = trainControl(method = "cv", number = 5),
  verbose = FALSE
)

# Predict and evaluate
predictions_2 <- predict(gbm_model_2, test_data_2)
mse_2 <- mean((predictions_2 - test_data_2$price)^2)
r2_2 <- cor(predictions_2, test_data_2$price)^2

# Print results
cat("Regression Task 1: Predict Flight Duration\n")
cat("MSE:", mse_1, "\n")
cat("R-squared:", r2_1, "\n\n")

cat("Regression Task 2: Predict Ticket Price\n")
cat("MSE:", mse_2, "\n")
cat("R-squared:", r2_2, "\n")




  
  

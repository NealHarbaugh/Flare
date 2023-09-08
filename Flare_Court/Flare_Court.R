

###-------User input------------------------------------------------------------

model_path <- "~/path/to/models"

path_to_input_data <- "~/path/to/data.csv"
path_to_output_data <- "~/path/to/write/data.csv"

# Indicate rounding for each column of data
Column_1_rounding <- 0
Column_2_rounding <- 0



###-------Load required libraries-----------------------------------------------

if (!require(dplyr)) {
  install.packages("dplyr")
  library(dplyr)
} else {
  library(dplyr)
}

if (!require(keras)) {
  install.packages("keras")
  library(keras)
} else {
  library(keras)
}

if (!require(tensorflow)) {
  install.packages("tensorflow")
  library(tensorflow)
} else {
  library(tensorflow)
}

if (!require(progress)) {
  install.packages("progress")
  library(progress)
} else {
  library(progress)
}

if (!require(stringr)) {
  install.packages("stringr")
  library(stringr)
} else {
  library(stringr)
}


###-------Create needed functions-----------------------------------------------

# Function to check if a column contains non-numeric characters
# Arguments:
#   column: A vector or column to check for non-numeric characters
# Returns:
#   TRUE if the column contains non-numeric characters, otherwise FALSE

check_column_for_letters <- function(column) {
  # Use sapply to iterate over values in the column and check if they are 
  # characters and not matching the numeric pattern
  result <- any(sapply(column, function(value) {
    is.character(value) && !grepl("^[0-9]+$", value)
  }))
  return(result)
}



# Function to re-encode a column in a data frame based on a codex
# Arguments:
#   data: The data frame to be re-encoded
#   codex: The codex data frame containing mapping information
#   col_name: The name of the column to re-encode
# Returns:
#   The data frame with the specified column re-encoded based on the codex

reencode_column <- function(data, codex, col_name) {
  # Get unique values and corresponding codes from the codex
  unique_values <- codex$Category[codex$Column == col_name]
  codes <- codex$Code[codex$Column == col_name]
  
  # Create a mapping dictionary
  mapping_dict <- setNames(codes, unique_values)
  
  # Apply the mapping to the specified column in the data frame
  data[[col_name]] <- mapping_dict[data[[col_name]]]
  
  # Return the data frame with the re-encoded column
  return(data)
}



# Function to round numeric columns in a data frame to the specified decimal 
# places
# Arguments:
#   data: The input data frame
#   decimal_places: A list or vector containing the number of decimal places 
#   for each column
# Returns:
#   The data frame with rounded numeric columns

adjust_decimal_places <- function(data, decimal_places) {
  
  # Get the column names of the data frame
  original_column_names <- colnames(data)
  if (is.null(original_column_names)) {
    Altered_column_names <- paste0("V", seq_len(ncol(data)))
    colnames(data) <- Altered_column_names
  }
  
  # Get the column names of the data frame
  Data_column_names <- colnames(data)
  # Get the total number of columns in the data frame
  Number_of_columns <- ncol(data)
  
  # Get the column names of the data frame
  Data_column_names <- colnames(data)
  
  # Loop through each column in the data frame
  for (i in 1:Number_of_columns) {
    
    # Get the name of the column to round
    col_to_round <- Data_column_names[i]
    
    # Find the index of the column in the data frame
    column_index <- colnames(data) == col_to_round
    
    # Round the values in the column to the specified decimal places
    data[, column_index] <- round(data[, column_index], decimal_places[i])
  }
  
  if (is.null(original_column_names)) {
    colnames(data) <- NULL
  }
  
  # Return the modified data frame with rounded numeric columns
  return(data)
}



# Function for model prediction and weighted voting
# Arguments:
#   First_model_vote: Predictions from the first model as a data frame
#   Second_model_vote: Predictions from the second model as a data frame
#   Third_model_vote: Predictions from the third model as a data frame
#   weights: Vector of weights for each model's predictions
# Returns:
#   Weighted ensemble predictions as a data frame

weighted_model_voting <- function(First_model_vote, Second_model_vote, 
                                  Third_model_vote, weights) {
  
  # Get the number of columns in the prediction data frames
  X <- ncol(First_model_vote)
  
  # Initialize an empty data frame
  blank_df <- data.frame(matrix(ncol = 0, nrow = 1))
  
  # Iterate through columns
  for (i in 1:X)
  {
    # Generate a unique column name
    column_name <- paste0("Column_", i)
    
    # Get predictions for each model
    Model_one_column_value <- First_model_vote[, i]
    Model_two_column_value <- Second_model_vote[, i]
    Model_three_column_value <- Third_model_vote[, i]
    
    # Calculate the weighted average of predictions for this column
    column_value <- ((Model_one_column_value * weights[1]) + 
                       (Model_two_column_value * weights[2]) + 
                       (Model_three_column_value * weights[3])) /
      sum(weights)
    
    # Add the column with the weighted predictions to the blank data frame
    blank_df[[column_name]] <- column_value 
    
  }
  
  # Combine the final predictions into a data frame
  final_df <- data.frame(blank_df) 
  
  # Reset column names to NULL
  colnames(final_df) <- NULL
  
  return(final_df)
}



# Function to decode a column in a data frame based on a codex
# Arguments:
#   data: The data frame to be decoded
#   codex: The codex data frame containing mapping information
#   col_name: The name of the column to decode
# Returns:
#   The data frame with the specified column decoded based on the codex

decode_column <- function(data, codex, col_name) {
  # Get unique values and corresponding codes from the codex
  codes <- codex$Code[codex$Column == col_name]
  unique_values <- codex$Category[codex$Column == col_name]
  
  # Create a mapping dictionary
  mapping_dict <- setNames(unique_values, codes)
  
  # Apply the mapping to the specified column in the data frame
  data[[col_name]] <- mapping_dict[data[[col_name]]]
  
  # Return the data frame with the decoded column
  return(data)
}


###-------Load data and models--------------------------------------------------

temp_folder <- paste0(model_path, "/Temporary_folder")

Codex_path <- paste0(temp_folder, "/Codex.csv")

Codex_df <- read.csv(Codex_path)

# Set model paths
Ignis_model_path <- paste0(model_path, "/1/Ignis_model.h5")
Pyra_model_path <- paste0(model_path, "/2/Pyra_model.h5")
Scorch_model_path <- paste0(model_path, "/3/Scorch_model.h5")

# Load models
Ignis_model <- load_model_hdf5(Ignis_model_path, compile = TRUE)
Pyra_model <- load_model_hdf5(Pyra_model_path, compile = TRUE)
Scorch_model <- load_model_hdf5(Scorch_model_path, compile = TRUE)

Leaderboard_path <- paste0(model_path, "/top_losses.csv")

# Load the leaderboard file with batch size values corresponding to the model
leaderboard <- read.csv(Leaderboard_path)

# Initialize variables to store model batch size values
model_batch_size <- numeric(3)

# Store the values for batch size
model_batch_size[1] <- leaderboard$Batch_Size[1]
model_batch_size[2] <- leaderboard$Batch_Size[2]
model_batch_size[3] <- leaderboard$Batch_Size[3]

# Load data 
input_data <- read.csv(path_to_input_data)

input_data <- input_data %>% select(-X) 

input_data$dominance <- str_to_title(input_data$dominance)

# Remove rows with missing data in preburn_data
input_data_processed <- na.omit(input_data)

# Apply the function to all columns 
columns_with_letters <- sapply(input_data_processed, check_column_for_letters)

# Get the names of columns with non-numeric characters
columns_with_letters <- names(columns_with_letters[columns_with_letters])

for (col in 1:length(columns_with_letters))
{
  col_name <- columns_with_letters[col]
  
  input_data_processed <- reencode_column(input_data_processed, Codex_df, 
                                          col_name)
}

# Pull the column names as a character vector
Data_column_names <- colnames(input_data_processed)

# Remove the column names temporarily for use later
colnames(input_data_processed) = NULL


###-------Run model predictions----------------------------------------------

# Create a list for rounding each column 
Rounding_list <- c(C1 = Column_1_rounding, C2 = Column_2_rounding, 
                   C3 = Column_3_rounding, C4 = Column_4_rounding, 
                   C5 = Column_5_rounding, C6 = Column_6_rounding, 
                   C7 = Column_7_rounding, C8 = Column_8_rounding)

Ignis_predictions <- data.frame(matrix(nrow = nrow(input_data_processed), 
                                       ncol = ncol(input_data_processed)))
Pyra_predictions <- data.frame(matrix(nrow = nrow(input_data_processed), 
                                      ncol = ncol(input_data_processed)))
Scorch_predictions <- data.frame(matrix(nrow = nrow(input_data_processed), 
                                        ncol = ncol(input_data_processed)))
colnames(Ignis_predictions) <- NULL
colnames(Pyra_predictions) <- NULL
colnames(Scorch_predictions) <- NULL

Iterations <- nrow(input_data_processed)
Ignis_iterations <- ceiling(Iterations/model_batch_size[1])
Pyra_iterations <- ceiling(Iterations/model_batch_size[2])
Scorch_iterations <- ceiling(Iterations/model_batch_size[3])

# Run Ignis' predictions for the input data 
cat("Running predictions for the Ignis model using the available data \n")
pb <- progress_bar$new(
  total = Ignis_iterations,
  format = paste(":current/:total [:bar] ETA: :eta ", sep = ""), 
  clear = FALSE)

for (i in 1:Ignis_iterations) {
  # Determine the current batch indices
  start_idx <- (i - 1) * model_batch_size[1] + 1
  end_idx <- min(i * model_batch_size[1], Iterations)
  
  # Get the corresponding batch of data
  input_batch <- input_data_processed[start_idx:end_idx, ]
  
  input_batch_matrix <- as.matrix(input_batch)
  
  # Predict for the batch
  pred_Ignis_batch <- predict(Ignis_model, input_batch_matrix, 
                              batch_size = model_batch_size[1], verbose = 0)
  
  # Store the predictions in the appropriate rows of the output data frame
  Ignis_predictions[start_idx:end_idx, ] <- pred_Ignis_batch
  
  pb$tick()
}

pb$terminate()


# Run Pyra's predictions for the input data 
cat("Running predictions for the Pyra model using the available data \n")
pb <- progress_bar$new(
  total = Pyra_iterations,
  format = paste(":current/:total [:bar] ETA: :eta ", sep = ""), 
  clear = FALSE)

for (i in 1:Pyra_iterations) {
  # Determine the current batch indices
  start_idx <- (i - 1) * model_batch_size[2] + 1
  end_idx <- min(i * model_batch_size[2], Iterations)
  
  # Get the corresponding batch of data
  input_batch <- input_data_processed[start_idx:end_idx, ]
  
  input_batch_matrix <- as.matrix(input_batch)
  
  # Predict for the batch
  pred_Pyra_batch <- predict(Pyra_model, input_batch_matrix, 
                             batch_size = model_batch_size[2], verbose = 0)
  
  # Store the predictions in the appropriate rows of the output data frame
  Pyra_predictions[start_idx:end_idx, ] <- pred_Pyra_batch
  
  pb$tick()
}
pb$terminate()


# Run Scorch's predictions for the input data 
cat("Running predictions for the Scorch model using the available data \n")
pb <- progress_bar$new(
  total = Scorch_iterations,
  format = paste(":current/:total [:bar] ETA: :eta ", sep = ""), 
  clear = FALSE)

for (i in 1:Scorch_iterations) {
  # Determine the current batch indices
  start_idx <- (i - 1) * model_batch_size[3] + 1
  end_idx <- min(i * model_batch_size[3], Iterations)
  
  # Get the corresponding batch of data
  input_batch <- input_data_processed[start_idx:end_idx, ]
  
  input_batch_matrix <- as.matrix(input_batch)
  
  # Predict for the batch
  pred_Scorch_batch <- predict(Scorch_model, input_batch_matrix, 
                               batch_size = model_batch_size[3], verbose = 0)
  
  # Store the predictions in the appropriate rows of the output data frame
  Scorch_predictions[start_idx:end_idx, ] <- pred_Scorch_batch
  
  pb$tick()
}
pb$terminate()

# Adjust decomal places of model predictions 
Ignis_predictions <- adjust_decimal_places(Ignis_predictions, Rounding_list)
Pyra_predictions<- adjust_decimal_places(Pyra_predictions, Rounding_list)
Scorch_predictions <- adjust_decimal_places(Scorch_predictions, Rounding_list)


###-------Prepare final output--------------------------------------------------

# Create a data frame to hold the finalized predictions
Final_predictions <- data.frame(matrix(nrow = nrow(input_data_processed), 
                                       ncol = ncol(input_data_processed)))

colnames(Final_predictions) <- NULL

# Define the weights based on model accuracy
weights <- c(3, 2, 1)  # Higher weight for Ignis, medium weight for Pyra, 
#                        and lowest weight for Scorch

pb <- progress_bar$new(total = Iterations, 
                       format = paste(":current/:total [:bar] ETA: :eta ", 
                                      sep = ""), clear = FALSE)

for (i in 1:Iterations) {
  Full_consensus <- FALSE
  Ignis_Pyra_consensus <- FALSE
  Ignis_Scorch_consensus <- FALSE
  Pyra_Scorch_consensus <- FALSE
  
  Ignis_vote <- Ignis_predictions[i, ]
  Pyra_vote <- Pyra_predictions[i, ]
  Scorch_vote <- Scorch_predictions[i, ]
  
  if (identical(Ignis_vote, Pyra_vote) && identical(Ignis_vote, Scorch_vote) && 
      identical(Pyra_vote, Scorch_vote)) {
    Full_consensus <- TRUE
  } else if (identical(Ignis_vote, Pyra_vote)) {
    Ignis_Pyra_consensus <- TRUE
  } else if (identical(Ignis_vote, Scorch_vote)) {
    Ignis_Scorch_consensus <- TRUE
  } else if (identical(Pyra_vote, Scorch_vote)) {
    Pyra_Scorch_consensus <- TRUE
  }
  
  if (Full_consensus == TRUE) {
    # Store the predictions in the appropriate rows of the output data frame
    Final_predictions[i, ] <- Ignis_vote
  } else if (Ignis_Pyra_consensus == TRUE || Ignis_Scorch_consensus == TRUE) {
    Final_predictions[i, ] <- Ignis_vote
  } else if (Pyra_Scorch_consensus == TRUE) {
    Final_predictions[i, ] <- Pyra_vote
  } else {
    Tribunal_vote <- weighted_model_voting(Ignis_vote, Pyra_vote, Scorch_vote, 
                                           weights)
    
    Final_predictions[i, ] <- Tribunal_vote
  }
  pb$tick()
}
pb$terminate()

# Round predictions
Output_predictions <- adjust_decimal_places(Final_predictions, Rounding_list)

# Reassign column names
colnames(Output_predictions) <- Data_column_names

# Turn categorical variables categorical again. 
for (col in 1:length(columns_with_letters))
{
  col_name <- columns_with_letters[col]
  
  Output_predictions <- decode_column(Output_predictions, Codex_df, col_name)
}


###-------Save the predictions to the output file-------------------------------

write.csv(Output_predictions, file = path_to_output_data, row.names = FALSE)




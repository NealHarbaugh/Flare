


###-------User input------------------------------------------------------------

# Write the path to your data sets
path_to_preburn_data <- "~/path/to/preburn/data.csv"
path_to_postburn_data <- "~/path/to/postburn/data.csv"

model_save_path <- "~/path/to/save/model/to"

batch_size <- 20

data_to_be_made_validation_data <- 0.1 # As a percent


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


###-------Process data----------------------------------------------------------

# Read the CSV file
preburn_data <- read.csv(path_to_preburn_data)
postburn_data <- read.csv(path_to_postburn_data)

# Remove rows with missing data in preburn_data
preburn_data_processed <- na.omit(preburn_data)

# Remove rows with missing data in postburn_data, matching rows in preburn_data
postburn_data_processed <- 
  na.omit(postburn_data[match(rownames(preburn_data_processed), 
                              rownames(postburn_data)),])

# Now, remove any remaining rows with missing data in postburn_data
postburn_data_processed <- na.omit(postburn_data_processed)

# Ensure the rows in preburn_data_processed and postburn_data_processed match
preburn_data_processed <- 
  preburn_data_processed[match(rownames(postburn_data_processed), 
                               rownames(preburn_data_processed)),]

# Apply the function to all columns 
columns_with_letters <- sapply(preburn_data_processed, check_column_for_letters)

# Get the names of columns with non-numeric characters
columns_with_letters <- names(columns_with_letters[columns_with_letters])

Codex_df <- data.frame(Category = character(0), 
                       Code = numeric(0),
                       Column = character(0),
                       Index = numeric(0))

for (col in 1:length(columns_with_letters))
{
  
  # Get the current column name
  col_name <- columns_with_letters[col]
  
  # Get unique values from preburn_data_processed
  unique_preburn <- unique(preburn_data_processed[[col_name]])
  
  # Get unique values from postburn_data_processed
  unique_postburn <- unique(postburn_data_processed[[col_name]])
  
  # Combine unique values from both datasets
  all_unique_values <- unique(c(unique_preburn, unique_postburn))
  
  # Assign codes based on the combined unique values
  key <- 1:length(all_unique_values)
  
  # Create a data frame to store the key for this column
  key_df <- data.frame(Category = all_unique_values, Code = key, 
                       Column = col_name)
  
  # Create a mapping based on the combined unique values
  categorical_mapping <- setNames(1:length(all_unique_values), 
                                  all_unique_values)
  
  # Apply the mapping to both data frames
  preburn_data_processed[[col_name]] <- 
    categorical_mapping[preburn_data_processed[[col_name]]]
  postburn_data_processed[[col_name]] <- 
    categorical_mapping[postburn_data_processed[[col_name]]]
  
  Codex_df <- rbind(Codex_df, key_df)
}


###-------Obtain the validation data--------------------------------------------

# Calculate the number of rows for validation in both datasets
num_rows_validation <- ceiling(nrow(preburn_data_processed) * 
                                 data_to_be_made_validation_data)

# Randomly select rows for validation from preburn_data_processed
validation_indices_preburn <- sample(1:nrow(preburn_data_processed), 
                                     num_rows_validation)

# Create the validation data for preburn_data_processed
validation_data_preburn <- preburn_data_processed[validation_indices_preburn, ]

# Remove the selected rows from preburn_data_processed
preburn_data_processed <- preburn_data_processed[-validation_indices_preburn, ]

# Match the validation rows in postburn_data_processed
validation_data_postburn <- 
  postburn_data_processed[match(rownames(validation_data_preburn), 
                                rownames(postburn_data_processed)), ]

# Remove the selected rows from postburn_data_processed
postburn_data_processed <- 
  postburn_data_processed[-match(rownames(validation_data_preburn), 
                                 rownames(postburn_data_processed)), ]

Validation_df <- data.frame(Index = validation_indices_preburn)

# Create a temporary folder to store hyperparameters
temp_folder <- paste0(model_save_path, "/Temporary_folder")
dir.create(temp_folder)

Codex <- paste0(temp_folder, "/Codex.csv")

Validation <- paste0(temp_folder, "/Validation.csv")

write.csv(Codex_df, Codex)

write.csv(Validation_df, Validation)

# Set the desired number of observations for each class
Ending_num_observations <- nrow(preburn_data_processed)


###-------Define model architecture---------------------------------------------

model_shape <- ncol(preburn_data_processed)

# Define the model architecture
Flare_model <- keras_model_sequential()
Flare_model %>%
  layer_reshape(target_shape = c(model_shape),
                input_shape = c(model_shape)) %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(
    units = 64,
    activation = "relu",
    kernel_regularizer = regularizer_l2(0.01)
  ) %>%
  layer_dense(units = model_shape)  # No activation function for regression

optimizer <- tf$keras$optimizers$Adam(learning_rate = 0.002)

# Compile the model with the appropriate loss function
Flare_model %>% compile(optimizer = optimizer,
                        loss = "mean_squared_error",
                        metrics = c("mse"))

# Save the new best model to a folder
save_model_hdf5(Flare_model,
                file.path(model_save_path, "/Flare_model.h5"),
                include_optimizer = TRUE)
model_path <- paste0(model_save_path, "/Flare_model.h5")


###-------Grid Search Hyperparameters-------------------------------------------

total_val_losses <- c()
total_val_accuracies <- c()

# Define the range of hyperparameters for the grid search
epochs_range <- seq(1, 41, by = 2)
batch_size_range <- seq(10, 30, by = 2)

# Create a data frame with every combination of epochs_range and
# batch_size_range
hyperparameter_combinations <- expand.grid(epochs = epochs_range,
                                           batch_size = batch_size_range)

# Create a data frame to store hyperparameters
hyperparameters_df <- 
  data.frame(batch_size = numeric(nrow(hyperparameter_combinations)),
             training_epochs = numeric(nrow(hyperparameter_combinations)))

# Initialize variables to store best hyperparameters and their performance
best_epochs <- numeric(3)
best_batch_size <- numeric(3)
best_losses <- rep(Inf, 3)

# Define loss variables
First_best_loss <- Inf
Second_best_loss <- Inf
Third_best_loss <- Inf

# Define empty lists to store the trained Flare models
#1
Ignis_model <- list()
#2
Pyra_model <- list()
#3
Scorch_model <- list()

# Initialize an empty list to store hyperparameter combinations
hyperparameter_list <- list()

Testing_iterations <-
  (length(epochs_range) * length(batch_size_range))

# Rename the columns
colnames(hyperparameters_df) <- c("batch_size", "training_epochs")

# Create folders for each of the top 3
Folder_1 <- paste0(model_save_path, "/1")
dir.create(Folder_1)
Folder_2 <- paste0(model_save_path, "/2")
dir.create(Folder_2)
Folder_3 <- paste0(model_save_path, "/3")
dir.create(Folder_3)

# Create a temporary folder
temp_folder <- paste0(model_save_path, "/Temporary_file")

# Create the progress bar
pb <- progress_bar$new(
  total = Testing_iterations,
  format = paste(":current/:total [:bar] ETA: :eta ", sep = ""),
  clear = FALSE
)

# Perform the grid search
for (g in 1:Testing_iterations)
{
  # Get the row corresponding to the current iteration
  current_row <- hyperparameter_combinations[g,]
  
  # Extract the values for batch_size and epochs
  batch_size <- current_row$batch_size
  training_epochs <- current_row$epochs
  
  epochs <- ceiling((Ending_num_observations / batch_size))
  
  # Add the hyperparameter combination to the list
  hyperparameter_list <- c(hyperparameter_list,
                           list(
                             list(batch_size = batch_size,
                                  training_epochs = training_epochs)
                           ))
  
  # Print the current combination of hyperparameters being tested
  cat(
    "\n",
    "Testing hyperparameters: batch_size =",
    batch_size,
    ", epochs =",
    training_epochs,
    "\n"
  )
  
  Flare_model <- load_model_hdf5(model_path, compile = TRUE)
  
  # Train the model with batch training
  for (epoch in 1:epochs) {
    if (epoch < epochs)
    {
      # Calculate the start index for the current batch
      start_index <- (epoch - 1) * batch_size + 1
      end_index <- start_index + batch_size - 1
    }
    else if (epoch == epochs)
    {
      total_samples <- nrow(postburn_data_processed)
      full_batches <- ceiling(total_samples / batch_size) - 1
      remaining_samples <-
        total_samples - (full_batches * batch_size)
      
      # Calculate the start index for the remaining samples
      start_index <- (epoch - 1) * batch_size + 1
      end_index <- start_index + remaining_samples - 1
    }
    
    # Extract the current batch from the data
    batch_x <- preburn_data_processed[start_index:end_index,]
    batch_y <- postburn_data_processed[start_index:end_index,]
    
    postburn_data_processed_matrix <- postburn_data_processed
    preburn_data_processed_matrix <- preburn_data_processed
    
    # Add index numbers as a new column to postburn_data_processed_matrix
    postburn_data_processed_matrix$index <-
      1:nrow(postburn_data_processed_matrix)
    preburn_data_processed_matrix$index <-
      1:nrow(preburn_data_processed_matrix)
    
    # Convert the data to matrices
    preburn_data_matrix <- as.matrix(batch_x)
    postburn_data_matrix <- as.matrix(batch_y)
    
    batch_x <- preburn_data_matrix
    batch_y <- postburn_data_matrix
    
    colnames(batch_x) <- NULL
    colnames(batch_y) <- NULL
    
    # Train the model with the current batch
    current_history <- Flare_model %>% fit(
      x = batch_x,
      y = batch_y,
      epochs = training_epochs,
      batch_size = batch_size,
      verbose = 0
    )
    
    if (epoch < epochs) {
      if (epoch > 1)
      {
        # Delete the temporary folder and its contents
        unlink(temp_folder, recursive = TRUE)
      }
      
      # Create a temporary folder for model and prediction files
      dir.create(temp_folder,
                 recursive = TRUE,
                 showWarnings = FALSE)
      
      # Save the temporary model
      Flare_save_path <- file.path(temp_folder,
                                   paste0("/Flare_model_", epoch, ".h5"))
      save_model_hdf5(Flare_model, Flare_save_path)
      
    } else if (epoch == epochs) {
      # Delete the temporary folder and its contents
      unlink(temp_folder, recursive = TRUE)
    }
  }
  
  trained_validation_postburn <- validation_data_postburn
  
  trained_validation_preburn <- validation_data_preburn
  
  trained_validation_preburn_matrix <-
    as.matrix(trained_validation_preburn)
  trained_validation_postburn_matrix <-
    as.matrix(trained_validation_postburn)
  
  colnames(trained_validation_preburn_matrix) <- NULL
  colnames(trained_validation_postburn_matrix) <- NULL
  
  # Evaluate the loss on the full validation data after training
  validation_loss <- Flare_model %>% evaluate(
    x = trained_validation_preburn_matrix,
    y = trained_validation_postburn_matrix,
    batch_size = batch_size,
    verbose = 0
  )
  
  # Extract just the first value into a separate variable
  validation_loss <- unname(validation_loss[1])
  
  # Update the top 3 best loss values if a new value is better
  if (validation_loss < First_best_loss) {
    # Remove the existing model in Folder_3
    file.remove(list.files(Folder_3, full.names = TRUE))
    
    # Move the existing model in Folder_2 to Folder_3
    file.rename(from = Folder_2, to = Folder_3)
    
    # Move the existing model in Folder_1 to Folder_2
    file.rename(from = Folder_1, to = Folder_2)
    
    dir.create(Folder_1)
    
    # Save the new best model to Folder_1
    save_model_hdf5(Flare_model, file.path(Folder_1, "/Ignis_model.h5"))
    
    Third_best_loss <- Second_best_loss
    Second_best_loss <- First_best_loss
    First_best_loss <- validation_loss
    
    Ignis_path <- paste0(Folder_2, "/Ignis_model.h5")
    Ignis_to_Pyra_path <- paste0(Folder_2, "/Pyra_model.h5")
    file.rename(Ignis_path, Ignis_to_Pyra_path)
    
    Pyra_path <- paste0(Folder_3, "/Pyra_model.h5")
    Pyra_to_Scorch_path <- paste0(Folder_3, "/Scorch_model.h5")
    file.rename(Pyra_path, Pyra_to_Scorch_path)
    
    # Update the best batch sizes and epochs
    best_batch_size[3] <- best_batch_size[2]
    best_batch_size[2] <- best_batch_size[1]
    best_batch_size[1] <- batch_size
    
    best_epochs[3] <- best_epochs[2]
    best_epochs[2] <- best_epochs[1]
    best_epochs[1] <- training_epochs
    
    # Update the stored models
    Scorch_model <- Pyra_model
    Pyra_model <- Ignis_model
    Ignis_model <- Flare_model
    
  } else if (validation_loss[1] < Second_best_loss) {
    # Remove the existing model in Folder_3
    file.remove(list.files(Folder_3, full.names = TRUE))
    
    # Move the existing model in Folder_2 to Folder_3
    file.rename(from = Folder_2, to = Folder_3)
    
    # Save the new best model to Folder_2
    save_model_hdf5(Flare_model, file.path(Folder_2, "/Pyra_model.h5"))
    
    Third_best_loss <- Second_best_loss
    Second_best_loss <- validation_loss
    
    Pyra_path <- paste0(Folder_3, "/Pyra_model.h5")
    Pyra_to_Scorch_path <- paste0(Folder_3, "/Scorch_model.h5")
    file.rename(Pyra_path, Pyra_to_Scorch_path)
    
    # Update the best batch sizes and epochs
    best_batch_size[3] <- best_batch_size[2]
    best_batch_size[2] <- batch_size
    
    best_epochs[3] <- best_epochs[2]
    best_epochs[2] <- training_epochs
    
    # Update the stored models
    Pyra_model <- Scorch_model
    Scorch_model <- Flare_model
    
    Scorch_model <- Pyra_model
    Pyra_model <- Flare_model
    
  } else if (validation_loss[1] < Third_best_loss) {
    # Remove the existing model in Folder_3
    file.remove(list.files(Folder_3, full.names = TRUE))
    
    # Save the new best model to Folder_3
    save_model_hdf5(Flare_model, file.path(Folder_3, "/Scorch_model.h5"))
    
    Third_best_loss <- validation_loss
    
    # Update the best batch sizes and epochs
    best_batch_size[3] <- batch_size
    
    best_epochs[3] <- training_epochs
    
    # Update the stored models
    Scorch_model <- Flare_model
  }
  
  # Add the new variables to the existing hyperparameter_df data frame
  row_to_update <- g
  hyperparameters_df[row_to_update, "batch_size"] <- batch_size
  hyperparameters_df[row_to_update, "training_epochs"] <-
    training_epochs
  
  write.csv(
    hyperparameters_df,
    file.path(model_save_path, "/hyperparameter_combinations_used.csv"),
    row.names = FALSE
  )
  
  # Create a data frame to store the top 3 loss values and hyperparameters
  top_losses_df <- data.frame(
    Rank = c("Best", "Second Best", "Third Best"),
    Loss = c(First_best_loss, Second_best_loss, Third_best_loss),
    Batch_Size = c(best_batch_size[1], best_batch_size[2], best_batch_size[3]),
    Training_Epochs = c(best_epochs[1], best_epochs[2], best_epochs[3])
  )
  
  # Define the file path for saving the data frame
  losses_csv_path <- file.path(model_save_path, "/top_losses.csv")
  
  # Write the data frame to a CSV file
  write.csv(top_losses_df, file = losses_csv_path, row.names = FALSE)
  
  # Update the progress bar with the normalized loss value
  pb$tick()
}

pb$terminate()


###-------Remove the temp folders-----------------------------------------------

# Remove the temporary folder and its contents
unlink(temp_folder, recursive = TRUE)


###-------Indicate completion---------------------------------------------------

# Print the 3 best hyperparameter combinations and their corresponding losses
cat("\n3 Best Hyperparameter Combinations:\n")
for (i in 1:3) {
  cat("\nCombination", i, ":\n")
  cat("Epochs:", best_epochs[i], "\n")
  cat("Batch Size:", best_batch_size[i], "\n")
  cat("Mean Validation Loss:", best_losses[i], "\n")
}

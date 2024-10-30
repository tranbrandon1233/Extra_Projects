# Prompt the user to select a CSV file
file_path <- file.choose()

# Read the CSV file, selecting only the first two columns
data <- read.csv(file_path, colClasses = c("numeric", "numeric", "NULL"))

# Extract the first two columns as vectors
variable1 <- data[, 1]
variable2 <- data[, 2]

# Perform a correlation test
correlation_result <- cor.test(variable1, variable2)

# Print the correlation test results
print(correlation_result)

# Plot the data with a regression line
plot(variable1, variable2, 
     xlab = colnames(data)[1], 
     ylab = colnames(data)[2], 
     main = "Scatter Plot with Correlation")
abline(lm(variable2 ~ variable1), col = "red")

# Add the correlation coefficient and p-value to the plot
text(min(variable1), max(variable2), 
     paste("Correlation:", round(correlation_result$estimate, 2), 
           "\nP-value:", round(correlation_result$p.value, 3)), 
     adj = c(0, 1))

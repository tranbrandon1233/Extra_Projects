# Get the current time and store it in variable "a"
execute_process(
  COMMAND date +%s
  OUTPUT_VARIABLE a
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Display a message
message("Time captured in variable 'a': ${a}")

# Wait for a few seconds (for demonstration purposes)
execute_process(COMMAND sleep 3)

# Get the current time again and store it in variable "b"
execute_process(
  COMMAND date +%s
  OUTPUT_VARIABLE b
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Calculate the time difference
math(EXPR time_diff "${b} - ${a}")

# Display the time difference
message("Time difference between 'a' and 'b': ${time_diff} seconds")
# Load necessary libraries
library(tidyverse)
library(dplyr)
library(knitr)
library(kableExtra)

# Generate sample data
n <- 1000 # Number of subjects


dm <- data.frame(
  USUBJID = 1:n,
  DOMAIN = "DM",
  ARM = sample(c("Placebo", "Drug A", "Drug B"), n, replace = TRUE),
  RFSTDTC = as.Date("2023-01-01") + sample(0:30, n, replace = TRUE),
  stringsAsFactors = FALSE
) %>%
  mutate(
    # Simulate study completion and discontinuation
    status = sample(c("Completed", "Discontinued AE", "Discontinued LOE","Discontinued Other", "Lost to Follow-up", "Other"), 
                    n, replace = TRUE, prob = c(0.7, 0.1, 0.05,0.05, 0.05, 0.05)),
    RFENDTC = case_when(
      status == "Completed" ~ RFSTDTC,
      status %in% c("Discontinued AE", "Discontinued LOE") ~ RFSTDTC + sample(1:89, n, replace = TRUE),
      status %in% c("Lost to Follow-up","Discounted Other","Other") ~ as.Date(NA)
    ),
    RFXSTDTC = case_when(
      status == "Completed" ~ as.Date("2023-01-01"),
      status %in% c("Discontinued AE", "Discontinued LOE", "Discounted Other") ~ RFENDTC,
      status == "Lost to Follow-up" ~ as.Date(NA),
      status == "Other" ~ RFSTDTC + sample(1:89, n, replace = TRUE)
    ),
    RFREASCD = case_when(
      status == "Discounted Other" ~ "Other",
      status == "Completed" ~ "",
      status == "Discontinued AE" ~ "AE",
      status == "Discontinued LOE" ~ "LOE",
      status == "Lost to Follow-up" ~ "LOST",
      status == "Other" ~ "OTHER"
    )
  ) %>%
  mutate(across(where(is.Date), ~if_else(is.na(.), "", as.character(.))))

print(head(dm))
# Create the Disposition Table
disposition_table <- dm %>%
  filter(DOMAIN == "DM") %>%
  group_by(ARM) %>%
  summarize(
    "Enrolled" = n(),
    "Completed" = sum(RFSTDTC == RFENDTC),
    "Discontinued" = sum(RFSTDTC != RFENDTC),
    "Discontinued due to AE" = sum(RFENDTC != "" & RFXSTDTC != "" & RFREASCD == "AE"),
    "Discontinued due to Lack of Efficacy" = sum(RFENDTC != "" & RFXSTDTC != "" & RFREASCD == "LOE"),
    "Discontinued due to Other Reasons" = sum(RFENDTC != "" & RFXSTDTC != "" & RFREASCD != "AE" & RFREASCD != "LOE"),
    "Lost to Follow-up" = sum(RFENDTC == "" & RFXSTDTC == ""),
    "Other" = sum(RFENDTC == "" & RFXSTDTC != "") 
  ) %>%
  ungroup() %>%
  # Transpose the table for presentation
  pivot_longer(cols = -ARM, names_to = "Category", values_to = "Count") %>%
  # Add percentages
  group_by(ARM) %>%
  mutate(Percentage = paste0(round(Count / sum(Count) * 100, 1), "%")) %>%
  ungroup() %>%
  # Reorder columns for presentation
  select(ARM, Category, Count, Percentage)

# Print the table
print(disposition_table,n=25)

# Export the table to an external file (optional)
write.csv(disposition_table, "disposition_table.csv", row.names = FALSE)
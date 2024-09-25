import csv
def process_csv(filename, new_filename):
    # Open the CSV file and read the data
    with open(filename, 'r') as csv_file:
        reader = csv.reader(csv_file)
        # Get the header
        header = next(reader)
        data = []
        for i in range(len(header)):
            header[i] = header[i].upper()  # Capitalize all characters in header
        data.append(header)  # Capitalize all characters in header  
        for row in reader:
            processed_row = [
                row[0].upper(),  # Capitalize all characters in name
                str(round(float(row[1]))),  # Round up the number
                row[2].upper()  # Capitalize all characters in city
            ]
            # Append the processed row to the data list
            data.append(processed_row)
    # Write the processed data to a new CSV file
    with open(new_filename, 'w', newline='') as new_file:
        writer = csv.writer(new_file)
        writer.writerows(data)

# Test the function
filename = "data.csv"
new_filename = "new_data.csv"
process_csv(filename, new_filename)
print(f"Processed {filename}. All characters capitalized and numbers rounded up.")
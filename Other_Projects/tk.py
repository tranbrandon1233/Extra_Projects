
from tkinter import messagebox, PhotoImage  # Import PhotoImage
import hashlib
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import os, re
import pandas as pd
from tkinter import messagebox
from datetime import datetime
from tkinter import messagebox, PhotoImage  # Import PhotoImage

class App(tk.Tk):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.title("Modern Bank Records")
        self.geometry("1368x600")
        self.style = ttk.Style()
        self.style.theme_use("clam")
        self.tree = None 
        # Initialize variables that do not depend on the UI
        self.flat_categories = [] 
        self.after_job = None
        self.last_id = 0
        self.uncategorized_var = tk.BooleanVar(value=False)
        self.categorized_count_str = tk.StringVar(value="Categorized Records: 0 (0%)")
        self.uncategorized_count_str = tk.StringVar(value="Uncategorized Records: 0 (0%)")
        self.total_records_str = tk.StringVar(value="Total Records: 0")
        self.default_view = True
        self.database_df = pd.DataFrame()
        self.search_var = None  # This is the search variable used throughout the class

        # Initialize UI components
        self.create_widgets()

        # Load or initialize the database after UI components that might use it are ready
        self.database_filename = self.generate_database_filename()
        print(f"Attempting to fetch database file: {self.database_filename}")
        self.initialize_or_load_database()
        self.update_statistics()

        # Now it's safe to refresh or manipulate the tree view
        self.filter_tree_view(self.search_var)
        
    def get_next_record_number(self):
        if self.database_df.empty:
            return 1  # Start from 1 if the database is empty
        else:
            return self.database_df['Record'].max() + 1

    def generate_record_id(self, record):
        """Generate an MD5 hash for a given record."""
        record_str = ','.join(map(str, record))  # Convert all items to string and concatenate
        return hashlib.md5(record_str.encode()).hexdigest()

    def generate_database_filename(self):
        # Generate MD5 hash of the current Python file for dynamic naming
        file_hash = self.generate_md5_hash_of_current_file()
        return f'database_{file_hash}.csv'

    def generate_md5_hash_of_current_file(self):
        with open(__file__, "rb") as file:
            file_content = file.read()
        return hashlib.md5(file_content).hexdigest()

    def initialize_or_load_database(self):
        self.database_filename = self.generate_database_filename()
        if os.path.exists(self.database_filename):
            self.database_df = pd.read_csv(self.database_filename)
            self.database_df['Record'] = pd.to_numeric(self.database_df['Record'], errors='coerce')
            print(f"Loaded existing database with {len(self.database_df)} records.")
        else:
            # Define DataFrame columns here or in a separate method if not already defined
            self.database_df = pd.DataFrame(columns=["Record", "Date Purchased", "Date Committed", "Figure", "Description", "Balance", "Category", "Sub-Category", "ID", "BANK"])
            self.database_df.to_csv(self.database_filename, index=False)
            print("Database file not found. Created a new one.")

    def transform_record_based_on_bank(self, index, row, bank_name, record_id):
        """Transforms the record based on the bank name and returns the transformed record.
        This method assumes transformation doesn't proceed if ID is duplicate."""
        if bank_name == "CommonWealth":
            return self.transform_commonwealth_record(index, row, record_id)
        # Add more conditions for other banks as necessary
        return None

    def add_record_to_database(self, record):
        """Add the provided record to the database if it's not a duplicate."""
        if not self.is_duplicate_id(record['ID']):
            self.database_df = pd.concat([self.database_df, pd.DataFrame([record])], ignore_index=True)
            self.database_df.to_csv(self.database_filename, index=False)
            print(f"Record added with ID: {record['ID']}")
        else:
            print(f"Duplicate record skipped with ID: {record['ID']}")
            
    def generate_md5_hash(value):
        """Generate MD5 hash for a given value."""
        return hashlib.md5(value.encode()).hexdigest()

    def categorize_record(self):
        selected_items = self.tree.selection()
        category = self.category_combobox.get()

        if not selected_items or not category:
            messagebox.showerror("Error", "Please select at least one record and a category.")
            print("Error: No record or category selected.")
            return

        # Assuming the category string is in the format "Category > Sub-Category"
        category_split = category.split(" > ")
        if len(category_split) != 2:
            messagebox.showerror("Error", "Invalid category selection.")
            print("Error: Invalid category selection.")
            return

        category_name, subcategory_name = category_split

        for item in selected_items:
            # Get the record ID from the selected item in the Tree View
            record_id = self.tree.item(item, 'values')[8]  # Assuming ID is at index 8

            # Update the DataFrame
            self.database_df.loc[self.database_df['ID'] == record_id, ['Category', 'Sub-Category']] = [category_name, subcategory_name]
        
        self.database_df.to_csv(self.database_filename, index=False)  # Save changes to the file
        self.filter_tree_view(self.search_var)  # Refresh the Tree View to reflect the changes
        print(f"Categorized {len(selected_items)} records as {category_name} > {subcategory_name}.")

    def load_records(self):
        print("Loading records...")

        filepath = filedialog.askopenfilename(
            title="Select a bank record file",
            filetypes=(("CSV files", "*.csv"), ("All files", "*.*"))
        )

        if not filepath:
            return

        try:
            input_df = pd.read_csv(filepath, header=None)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load the file: {e}")
            return

        new_records_count = 0
        duplicate_records_count = 0
        for index, row in input_df.iterrows():
            try:
                row_list = row.tolist()
                bank_name = self.detect_bank_pattern(row_list)
                if bank_name:
                    record_id = self.generate_record_id(row_list)
                    # Assume transformation methods are correctly defined elsewhere
                    new_record = self.transform_commonwealth_record(index, row_list, self.generate_md5_hash_of_current_file(), record_id)
                    if self.add_record_if_unique(new_record):
                        new_records_count += 1
                    else:
                        duplicate_records_count += 1
            except Exception as e:
                print(f"Error processing record at index {index}: {e}")

        # Save the DataFrame to CSV after processing all new records
        if new_records_count > 0 or duplicate_records_count > 0:
            self.database_df.to_csv(self.database_filename, index=False)
            print(f"Database updated. {new_records_count} new records added. {duplicate_records_count} duplicates found and skipped.")

            # Prepare the message
            message = f"Import Summary:\n\n- Number of records added: {new_records_count}\n- Number of duplicate records: {duplicate_records_count}"
            messagebox.showwarning("Import Summary", message)  # Display the warning message with the stats
        else:
            messagebox.showinfo("Import Summary", "No new records were added.")

        self.filter_tree_view(self.search_var)  # Refresh the Tree View to reflect the changes

    def show_chart(self):
        # Toggle to chart view if currently in default view
        if self.default_view:
            self.toggle_view()
        messagebox.showinfo("Info", "Chart display functionality goes here.")

    def create_widgets(self):
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

  
        # Create a search framework
        search_frame = ttk.Frame(main_frame)
        search_frame.pack(pady=10, fill=tk.X)

        # Create a search input box
        search_input = ttk.Entry(search_frame, textvariable=self.search_var, width=50)  
        search_input.grid(row=0, column=1, sticky="ew", padx=(5, 5))
        
        # Connect the search box to the filtering method
        
        buttons_frame = ttk.Frame(main_frame)
        buttons_frame.pack(pady=10, fill=tk.X)

        load_btn = ttk.Button(buttons_frame, text="Load Bank Records", command=self.load_records, style="TButton")
        load_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)

        show_chart_btn = ttk.Button(buttons_frame, text="Show Chart", command=self.show_chart, style="TButton")
        show_chart_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)

        search_frame = ttk.Frame(main_frame)
        search_frame.pack(pady=10, fill=tk.X)

        # delete_icon = PhotoImage(file='./lib/delete.png')  # Ensure the path to the icon is correct
        delete_icon = PhotoImage(file='delete.png')  # Ensure the path to the icon is correct

        search_var = tk.StringVar()
        search_input = ttk.Entry(search_frame, textvariable=search_var, width=50)  
        search_input.grid(row=0, column=1, sticky="ew", padx=(5, 5))
        search_var.trace_add("write", lambda name, index, mode, sv=search_var: self.filter_tree_view(sv))

        clear_search_btn = tk.Button(search_frame, image=delete_icon, width=14, height=14, command=lambda: search_var.set(''))
        clear_search_btn.image = delete_icon  
        clear_search_btn.grid(row=0, column=0, padx=(0, 5))  

        search_input.insert(0, "Search Records...")
        search_input.bind("<FocusIn>", lambda args: search_input.delete('0', 'end') if search_var.get() == "Search Records..." else None)
#        search_input.bind("<FocusOut>", lambda args: search_input.insert(0, "Search Records...") if not search_var.get() else None)

        uncategorized_checkbox = ttk.Checkbutton(search_frame, text="Show Uncategorized Records Only", variable=self.uncategorized_var, command=self.filter_tree_view)
        uncategorized_checkbox.grid(row=0, column=5, sticky="w")

        self.tree = ttk.Treeview(main_frame, columns=("Record", "Date Purchased", "Date Committed", "Figure", "Description", "Balance", "Category", "Sub-Category"), show="headings")
        column_widths = [5, 10, 10, 5, 70, 5, 5, 5]
        for col, width in zip(self.tree["columns"], column_widths):
            self.tree.heading(col, text=col, command=lambda _col=col: self.treeview_sort_column(self.tree, _col, False))
            self.tree.column(col, width=width * 8, anchor=tk.CENTER)
            self.tree.config(selectmode='extended')


        self.tree.pack(pady=10, fill=tk.BOTH, expand=True)

        # Changes start here
        category_action_frame = ttk.Frame(main_frame)
        category_action_frame.pack(pady=2, fill=tk.X)

        self.category_var = tk.StringVar()
        self.category_combobox = ttk.Combobox(category_action_frame, textvariable=self.category_var, style="TCombobox", width=20)
        self.category_combobox.grid(row=0, column=0, sticky="ew")
        categorize_button = ttk.Button(category_action_frame, text="Categorize Record", command=self.categorize_record)
        categorize_button.grid(row=0, column=1, padx=10)

        # Statistics section moved below
        self.stats_frame = ttk.Frame(main_frame)
        self.stats_frame.pack(pady=10, fill=tk.X)

        self.sum_label = ttk.Label(self.stats_frame, text="Sum of Selected Records: 0", font=('Calibri', 10, 'bold'))
        self.sum_label.pack(side=tk.LEFT, padx=(10, 2))

        self.categorized_label = ttk.Label(self.stats_frame, textvariable=self.categorized_count_str, font=('Calibri', 10, 'bold'))  # Add this line
        self.categorized_label.pack(side=tk.LEFT, padx=(10, 2))  # Add this line
        self.uncategorized_label = ttk.Label(self.stats_frame, textvariable=self.uncategorized_count_str, font=('Calibri', 10, 'bold'))
        self.uncategorized_label.pack(side=tk.LEFT, padx=(10, 2))

        self.total_records_label = ttk.Label(self.stats_frame, textvariable=self.total_records_str, font=('Calibri', 10, 'bold'))
        self.total_records_label.pack(side=tk.LEFT, padx=(10, 2))
        # Changes end here

        self.main_frame = main_frame
        self.tree = self.tree
        self.load_btn = load_btn  
        self.show_chart_btn = show_chart_btn  
        self.tree.bind('<<TreeviewSelect>>', self.update_sum_of_selected_records)

        self.category_combobox.bind('<KeyRelease>', self.filter_category_options)
        self.filter_category_options()
        # Create a checkbox to toggle the display of uncategorized records
        uncategorized_checkbox = ttk.Checkbutton(
          search_frame, 
          text="Show Uncategorized Records Only", 
          variable=self.uncategorized_var, 
          command=self.filter_tree_view)  # Function to call when check box state changes
        uncategorized_checkbox.grid(row=0, column=5, sticky="w")

    def update_statistics(self):
        # Calculate and update the statistics
        total_records = len(self.database_df)
        uncategorized_records = len(self.database_df[self.database_df['Category'] == ""])
        categorized_records = total_records - uncategorized_records  # Calculate categorized records
        uncategorized_percentage = (uncategorized_records / total_records * 100) if total_records > 0 else 0
        categorized_percentage = (categorized_records / total_records * 100) if total_records > 0 else 0  # Calculate percentage of categorized records

        self.uncategorized_count_str.set(f"Uncategorized Records: {uncategorized_records} ({uncategorized_percentage:.2f}%)")
        self.categorized_count_str.set(f"Categorized Records: {categorized_records} ({categorized_percentage:.2f}%)")  # Update categorized records string
        self.total_records_str.set(f"Total Records: {total_records}")

    def update_sum_of_selected_records(self, event):
        selected = self.tree.selection()  # Get the item IDs of the selected records
        sum_figure = 0
        count_selected = len(selected)  # Count of the selected records
        for item_id in selected:
            # Assuming the 'Figure' column is at index 3
            item_figure = self.tree.item(item_id, 'values')[3]
            try:
                sum_figure += float(item_figure.replace(',', ''))  # Remove commas and convert to float
            except ValueError:
                continue  # Skip items where conversion to float fails
        self.sum_label.config(text=f"Count: {count_selected}, Sum of Selected Records: {sum_figure:.2f}")


    def treeview_sort_column(self, tree, col, reverse):
        l = [(tree.set(k, col), k) for k in tree.get_children('')]
        
        # Adjust sorting logic based on column
        if col in ["Record", "Figure", "Balance"]:  # Assuming these are numeric
            try:
                l.sort(key=lambda t: float(t[0].replace(',', '')) if t[0] else float('-inf'), reverse=reverse)
            except ValueError:
                pass  # Handle case where conversion to float fails
        elif col in ["Date Purchased", "Date Committed"]:  # Assuming date format is DD/MM/YYYY
            try:
                l.sort(key=lambda t: datetime.strptime(t[0], '%d/%m/%Y') if t[0] else datetime.min, reverse=reverse)
            except ValueError:
                pass  # Handle case where conversion to datetime fails
        else:
            l.sort(reverse=reverse)  # Default to string sorting

        for index, (val, k) in enumerate(l):
            tree.move(k, '', index)

        # Reverse sorting order for next time
        tree.heading(col, command=lambda _col=col, _tree=tree: self.treeview_sort_column(_tree, _col, not reverse))

    def initialize_or_load_database(self):
        if os.path.exists(self.database_filename):
            self.database_df = pd.read_csv(self.database_filename).fillna('')
        else:
            self.database_df = pd.DataFrame(columns=[
                "Record", "Date Purchased", "Date Committed", "Figure", "Description", 
                "Balance", "Category", "Sub-Category", "ID", "BANK"
            ])
            self.database_df.to_csv(self.database_filename, index=False)
            print("Database file not found. Created a new one.")

        # Check for duplicate IDs after loading or creating the DataFrame
        if not self.database_df['ID'].is_unique:
            print("Warning: Duplicate IDs found in the DataFrame.")
        self.filter_tree_view(self.search_var)  # Filter the records to show only uncategorized records

    def read_categories(self, filepath):
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            headers = df.columns.tolist()
            categories = {header: [] for header in headers}
            for _, row in df.iterrows():
                for header in headers:
                    item = row[header]
                    if pd.notnull(item):  # Check if the cell is not empty
                        categories[header].append(item)
            # self.flat_categories = [f"{header} > {item}" for header, items in categories.items() for item in items]
            self.flat_categories = ["head > tail", "1 > 2"]

    def filter_tree_view(self,sv=None):
        # If the tree is empty, return
        if not self.tree:
            return
        # Clear the current view
        for item in self.tree.get_children():
            self.tree.delete(item)

        # Filter and display based on checkbox state
        if self.uncategorized_var.get():
            # Show only records where Category is blank
            filtered_df = self.database_df[self.database_df['Category'] == ""]
        else:
            # Show all records
            filtered_df = self.database_df
        
        # Applying the search filter
        if sv:
            self.search_var = sv
        if self.search_var:
            search_term = self.search_var.lower()
            filtered_df = filtered_df[self.database_df.apply(lambda row: search_term in str(row).lower(), axis=1)]
      
        for _, row in filtered_df.iterrows():
            self.tree.insert("", "end", values=(
                row["Record"],
                row["Date Purchased"],
                row["Date Committed"],
                row["Figure"],
                row["Description"],
                row["Balance"],
                row["Category"],
                row["Sub-Category"],
                row["ID"]
            ))
        self.update_statistics()

    def filter_category_options(self, event=None):
        # Cancel the existing job if it exists
        if self.after_job is not None:
            self.after_cancel(self.after_job)
            
        def update_combobox():
            # current_input = event.widget.get()  # Use the event widget to get current input
            filtered_options = [option for option in self.flat_categories ] #if current_input.lower() in option.lower()]
            
            self.category_combobox['values'] = filtered_options

            if filtered_options:
                # Open the dropdown if there are matching options
                self.category_combobox.event_generate('<Down>')
            else:
                # Close the dropdown if there are no matching options
                self.category_combobox.event_generate('<Escape>')

        # Schedule the update_combobox function to run after 200 milliseconds, which allows for real-time filtering
        self.after_job = self.after(200, update_combobox)

    def on_combobox_select(self, event=None):
        # This method is called when a selection is made from the combobox dropdown.
        # It ensures that after selecting an item, the focus returns to the input field, allowing for continuous typing.
        self.category_combobox.focus()

        # Optional: Place the cursor at the end of the input field to continue typing
        self.category_combobox.selection_clear()

    def add_record_if_unique(self, new_record):
        """Add a new record to the database if the ID is unique."""
        record_id = new_record['ID']
        if record_id not in self.database_df['ID'].values:
            self.database_df = pd.concat([self.database_df, pd.DataFrame([new_record])], ignore_index=True)
            print(f"Record with ID {record_id} added successfully.")
            return True
        else:
            print(f"Duplicate record with ID {record_id} found. Record not added.")
            return False

    def process_new_records(self, input_df):
        new_records_count = 0
        duplicate_records_count = 0
        for index, row in input_df.iterrows():
            bank_name = self.detect_bank_pattern(row.tolist())
            if bank_name:
                record_id = self.generate_record_id(row.tolist())
                if record_id not in self.database_df['ID'].values:
                    # Only add new records with unique IDs
                    transformed_record = self.transform_commonwealth_record(index, row.tolist(), record_id)  # Assuming transformation method signature is updated
                    self.database_df = pd.concat([self.database_df, pd.DataFrame([transformed_record])], ignore_index=True)
                    new_records_count += 1
                else:
                    duplicate_records_count += 1
        if new_records_count > 0:
            self.database_df.to_csv(self.database_filename, index=False)
            print(f"Added {new_records_count} new records to the database.")
        if duplicate_records_count > 0:
            print(f"Skipped {duplicate_records_count} duplicate records.")
        messagebox.showinfo("Import Summary", f"Total new records added: {new_records_count}\nTotal duplicate records skipped: {duplicate_records_count}")

    def detect_bank_pattern(self, row):
        try:
            if len(row) >= 4:  # Assuming at least 4 columns for CommonWealth
                # Add specific logic to confirm it matches the expected pattern
                return "CommonWealth"
            # You can add more patterns here
        except Exception as e:
            print(f"Error detecting bank pattern: {e}")
        return None

    def transform_commonwealth_record(self, index, row, hash, record_id):
        # Assuming row format might include date_purchased, figure, description, balance as a basic example
        date_purchased, figure, description, balance = row

        # Extract "Value Date" if present in the description for "Date Committed"
        value_date_search = re.search(r"Value Date: (\d{2}/\d{2}/\d{4})", description)
        # The value_date is intended for "Date Committed"
        value_date = value_date_search.group(1) if value_date_search else date_purchased  # Fallback to date_purchased if not found

        # Inverting the fields correctly based on the request
        transformed_record = {
            "Record": self.get_next_record_number(),
            "Date Purchased": value_date,  # Use extracted value_date or date_purchased for "Date Purchased"
            "Date Committed": date_purchased,  # Original date_purchased is now used for "Date Committed"
            "Figure": figure,
            "Description": description,
            "Balance": balance,
            "Category": "",
            "Sub-Category": "",
            "ID": record_id,
            "BANK": "CommonWealth"
        }

        print(f"Transformed record: {transformed_record}")
        return transformed_record
   
    def show_chart(self):
        # Toggle to chart view if currently in default view
        if self.default_view:
            self.toggle_view()
        else:
            # If not in the default view, this means we're currently showing the chart,
            # so toggle back to the default view (the else block is optional based on your requirements)
            self.toggle_view()

def main():
    app = App()
    app.read_categories('categories.csv')  # Assuming you have a categories.csv
    app.mainloop()

if __name__ == "__main__":
    main()
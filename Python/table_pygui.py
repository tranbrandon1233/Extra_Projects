import dearpygui.dearpygui as dpg

dpg.create_context()

# Sample data for the table
table_data = [
    ["Name", "Age", "City"],
    ["John Doe", 30, "New York"],
    ["Jane Doe", 25, "Los Angeles"],
    ["Peter Pan", 18, "Neverland"],
]

# Function to update the table data
def update_table(new_data):
    # Clear existing rows (except header)
    for row in range(1, len(table_data)):
        dpg.delete_item(f"tr_{row}")

    # Add new rows
    for row_index, row_data in enumerate(new_data):
        with dpg.table_row(tag=f"tr_{row_index+1}", parent="table"):
            for cell_data in row_data:
                dpg.add_text(cell_data)

# Create the table
with dpg.window(label="Table Example", width=500, height=300):
    with dpg.table(header_row=True, tag="table"):
        # Create header row
        with dpg.table_row():
            for header in table_data[0]:
                dpg.add_table_column(label=header)

        # Add initial data rows
        for row_index, row_data in enumerate(table_data[1:]):
            with dpg.table_row(tag=f"tr_{row_index+1}"):
                for cell_data in row_data:
                    dpg.add_text(cell_data)

# Example usage: Update the table with new data
new_data = [
    ["Alice", 28, "Chicago"],
    ["Bob", 35, "Seattle"],
]
update_table(new_data)

dpg.create_viewport(title='Dear PyGui Table', width=800, height=600)
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()
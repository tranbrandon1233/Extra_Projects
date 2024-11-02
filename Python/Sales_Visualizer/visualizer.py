import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import customtkinter as ctk
import tkinter as tk

class GroceryDataVisualization(ctk.CTkFrame):
    def __init__(self, parent):
        """
        Initialize the GroceryDataVisualization frame with scrollable matplotlib plots
        
        Args:
            parent: The parent window/widget this frame will be placed in
        """
        # Initialize the frame with the parent widget
        super().__init__(parent)
        
        # Configure grid weights for responsive layout
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        
        # Create scrollable canvas setup with a white background for better visibility
        self.canvas = tk.Canvas(self, bg='white')
        self.scrollbar = ctk.CTkScrollbar(
            self, 
            orientation="vertical", 
            command=self.canvas.yview
        )
        
        # Configure canvas scrolling
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.scrollable_frame = ctk.CTkFrame(self.canvas)
        self.canvas_frame = self.canvas.create_window(
            (0, 0), 
            window=self.scrollable_frame, 
            anchor="nw"
        )
        
        # Bind events for proper scrolling behavior
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")),
        )
        self.canvas.bind(
            "<Configure>",
            lambda e: self.canvas.itemconfig(self.canvas_frame, width=e.width),
        )
        
        # Enable mousewheel scrolling
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        
        # Layout the canvas and scrollbar
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.scrollbar.grid(row=0, column=1, sticky="ns")
        
        # Read and process data
        df = pd.read_csv('./data.csv')
        
        # Convert Date column to datetime if it exists
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        
        # Calculate various aggregations
        categorySales = df.groupby('Category')['Quantity'].sum() if all(col in df.columns for col in ['Category', 'Quantity']) else None
        employeeSales = df.groupby('Employee')['Total'].sum() if all(col in df.columns for col in ['Employee', 'Total']) else None
        dailySales = df.groupby('Date')['Total'].sum() if all(col in df.columns for col in ['Date', 'Total']) else None
        totalSales = df.groupby('Category')['Total'].sum() if all(col in df.columns for col in ['Category', 'Total']) else None
        
        # Create subplots with increased spacing
        fig, axs = plt.subplots(4, 1, figsize=(10, 16))
        plt.subplots_adjust(hspace=0.5)  # Increase vertical spacing between plots
        
        # Plot 1: Category Pie Chart
        if categorySales is not None:
            categorySales.plot.pie(ax=axs[0], autopct='%1.1f%%')
            axs[0].set_ylabel('')
            axs[0].set_title('Most and Least Categories', pad=20)
        
        # Plot 2: Daily Sales Line Chart
        if dailySales is not None:
            dailySales.plot(ax=axs[1], kind='line', marker='o')
            axs[1].set_xlabel('Date')
            axs[1].set_ylabel('Total Sales')
            axs[1].set_title('Total Sales Over Time', pad=20)
            axs[1].tick_params(axis='x', rotation=45)
        
        # Plot 3: Employee Sales Bar Chart
        if employeeSales is not None:
            employeeSales.plot.bar(ax=axs[2], color=['blue', 'green', 'yellow'])
            axs[2].set_xlabel('Employee')
            axs[2].set_ylabel('Total Sales')
            axs[2].set_title('Total Sales per Employee', pad=20)
            # Rotate x-axis labels to horizontal
            axs[2].tick_params(axis='x', rotation=0)
            # Adjust bottom margin to prevent label cutoff
            plt.setp(axs[2].get_xticklabels(), ha='center')
        
        # Plot 4: Category Sales Bar Chart
        if totalSales is not None:
            totalSales.plot.bar(ax=axs[3], color=['#1f77b4', '#ff7f0e', '#2ca02c', "#d62728", '#9467bd'])
            axs[3].set_xlabel('Category')
            axs[3].set_ylabel('Total Sales')
            axs[3].set_title('Total Sales by Category', pad=20)
            # Rotate x-axis labels to horizontal
            axs[3].tick_params(axis='x', rotation=0)
            # Adjust bottom margin to prevent label cutoff
            plt.setp(axs[3].get_xticklabels(), ha='center')
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        # Create and pack the matplotlib canvas
        canvas = FigureCanvasTkAgg(fig, master=self.scrollable_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)
    
    def _on_mousewheel(self, event):
        """
        Handle mousewheel scrolling
        
        Args:
            event: The mousewheel event
        """
        # Scroll up/down (-1/1) * scroll speed
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

if __name__ == "__main__":
    # Create the main application window
    app = ctk.CTk()
    app.geometry("800x600")
    app.grid_rowconfigure(0, weight=1)
    app.grid_columnconfigure(0, weight=1)
    app.title("Grocery Sales Dashboard")  # Added window title
    
    # Create and grid the visualization frame
    frame = GroceryDataVisualization(app)
    frame.grid(row=0, column=0, sticky="nsew")
    
    # Start the application
    app.mainloop()
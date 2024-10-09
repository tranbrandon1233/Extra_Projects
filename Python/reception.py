import streamlit as st
import graphviz
import pandas as pd
import random
from IPython.display import display, HTML

# Create a graph object
graph = graphviz.Digraph('hotel_reception_data_flow', format='png')

# Add nodes
nodes = ['Intake', 'Reception', 'Stay_Info', 'Billing', 'Keycard',
         'Room_Assign', 'Daily_Checklist', 'Supervisor', 'Daily_Events', 'Occupancy_Pred']

for node in nodes:
    graph.node(node, shape='box')

# Add edges
edges = [('Intake', 'Reception'), ('Reception', 'Stay_Info'), ('Stay_Info', 'Billing'), ('Billing', 'Keycard'),
         ('Keycard', 'Room_Assign'), ('Room_Assign', 'Daily_Checklist'), ('Daily_Checklist', 'Supervisor'), 
         ('Supervisor', 'Daily_Events'), ('Daily_Events', 'Occupancy_Pred'), ('Stay_Info', 'Daily_Checklist'),
         ('Billing', 'Daily_Events'), ('Room_Assign', 'Occupancy_Pred')]

for edge in edges:
    graph.edge(*edge)

# Render the graph
graph.view()

# Artificial data generation for each node
data = {}
for node in nodes:
    data[node] = pd.DataFrame({f"{node}_feature_{i}": [random.randint(0, 100) for _ in range(5)] for i in range(1, 6)})

# Interactive dropdown menu to show data
def view_data(node):
    display(HTML(data[node].to_html())) 
    
# Artificial data generation
data = {}
for node in nodes:
    data[node] = pd.DataFrame({f"{node}_feature_{i}": [random.randint(0, 100) for _ in range(5)] for i in range(1, 6)})

# Streamlit app
st.title("Hotel Reception Data Flow")

# Display the graph
st.graphviz_chart(graph)

# Dropdown menu
selected_node = st.selectbox("Select a node to view data:", nodes)

# Display data for the selected node
st.write(data[selected_node]) 
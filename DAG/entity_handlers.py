
# from dag import Entity
from entity import Entity  # Import Entity from entity.py

class DataRowStore(Entity):
    def __init__(self, data, name):
        super().__init__([])
        self.data = data
        self.name = name

    def recompute(self, data=None, namespace=None):
        if data is not None:
            updated_rows = set()
            for i in range(len(self.data.keys())): # Loop for every index in the data dictionary
                updated_rows.add(i) # Add the index to the updated rows set
            self.data = data                
            if namespace:
                namespace.graph["Total Compensation"].dirty_rows = updated_rows  # Track dirty rows
                self.push_dirt(namespace)      
        return self.data

    def push_dirt(self, namespace):
        namespace._propagate_dirt(self.name)

    def subscribe(self, component):
        pass


# Adjustments in MathColumn to handle data correctly
class MathColumn(Entity):
    def __init__(self, operation, input_columns, name):
        super().__init__([input_column.split('.')[0] for input_column in input_columns])
        self.operation = operation
        self.input_columns = input_columns
        self.name = name
        self.result = {}

    def recompute(self, data, namespace=None):
        # Ensure operation handling aligns with expected data structure
        if self.operation == "add":  # Check if the operation is addition
            result = {}
            entity1, _ = self.input_columns[0].split('.') # Get the entities of each input column
            entity2, _ = self.input_columns[1].split('.')
            column1Keys = list(data[entity1].keys())  # Get the keys of each column
            column2Keys = list(data[entity2].keys())
            for i in range(len(column1Keys)): # Iterate through the indices of the columns
                try:
                    result[i] = data[entity1][column1Keys[i]][i] + data[entity2][column2Keys[i]][i]   # Assign the result of the column addition to the result dictionary based on their corresponding indices as the keys
                except KeyError as e:
                    print(f"KeyError accessing data: {e}") # Removed the reference to the key variable since it is no longer used
            self.result = result        
            if namespace: 
                self.push_dirt(namespace)
            self.dirty_rows.update(set(result.keys()))  # Track dirty rows
        return self.result


    def push_dirt(self, namespace):
        namespace._propagate_dirt(self.name)

    def subscribe(self, component):
        pass
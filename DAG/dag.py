
from graphlib import TopologicalSorter
from entity_handlers import DataRowStore, MathColumn  # Import from entity_handlers.py

class Node:
    def __init__(self, name, entity_type, dependencies=None):
        self.name = name
        self.type = entity_type
        self.dependencies = dependencies or []
        self.result = None
        self.dirty = False
        self.dirty_rows = set()  # Track dirty rows for each entity
        self.uses_thunks = False  # Added for thunk support

class DAG:
    def __init__(self):
        self.graph = {}

    def add_entity(self, name, entity_type, dependencies=None):
        node = Node(name, entity_type, dependencies)
        self.graph[name] = node

    def get_dependencies(self, entity_name):
        # Get entities that depend on the given entity, not its dependencies
        return [name for name, node in self.graph.items() if entity_name in node.dependencies]

    def get_result(self, entity_name):
        entity = self.graph[entity_name]
        if entity.dirty or any(self.graph[dep].dirty for dep in entity.dependencies):
            self._recompute(entity_name)
        if entity.uses_thunks:  # Evaluate thunk if needed
            return entity.result.evaluate(self)
        else:
            return entity.result

    def generate_dot(self):
        dot_representation = "digraph DAG {\n"
        for node_name, node in self.graph.items():
            for dependency in node.dependencies:
                dot_representation += f'"{dependency}" -> "{node_name}";\n'
        dot_representation += "}"
        return dot_representation

    def _recompute(self, entity_name):
        entity = self.graph[entity_name]
        if entity_name=="Total Compensation": # Since the "Total Compensation" value has no data to recompute, we can skip the recompute step
            entity.dirty=False
            entity.dirty_rows.clear()
        dependencies = entity.dependencies 
        data = {dep: self.get_result(dep) for dep in dependencies}
        for dependency in dependencies: # Loop for every dependency
            if self.graph[dependency].dirty:  # Check if any dependency is still dirty
                self._recompute(dependency) # Recompute the dependency
                if entity.type == "DataRowStore":   # Create a handler based on the type of the entity
                    handler = DataRowStore(data, entity_name)
                elif entity.type == "MathColumn":
                    handler = MathColumn(dependency.operation, dependency.input_columns, dependency)  
                # Add more entity types as needed
                entity.result = handler.recompute(data, self) # Set the result of the entity to be the computed output of the data using the defined handler
        entity.dirty = False
        if entity.uses_thunks:  # Propagate thunk usage
            for dependent_name in self.get_dependencies(entity_name):
                self.graph[dependent_name].uses_thunks = True

    def _topological_sort(self):
        ts = TopologicalSorter(self.graph)
        return list(ts.static_order())

    def _propagate_dirt(self, entity_name):
        entity = self.graph[entity_name]
        for dependent_name in self.get_dependencies(entity_name):
            dependent = self.graph[dependent_name]
            dependent.dirty_rows.update(entity.dirty_rows)  # Update dirty rows of dependent
            dependent.dirty = True
            print(f"Propagating dirt to {dependent_name}")
        entity.dirty_rows.clear()
        print(f"Marked {entity_name} as dirty")
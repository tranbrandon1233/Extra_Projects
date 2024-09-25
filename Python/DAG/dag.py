from graphlib import TopologicalSorter
from entity_handlers import DataRowStore, MathColumn 

class Node:
    def __init__(self, name, entity_type, dependencies=None):
        self.name = name
        self.type = entity_type
        self.dependencies = dependencies or []
        self.result = None
        self.dirty = False
        '''
        Track dirty rows for each entity
        '''
        self.dirty_rows = set()  
        '''
        Added for thunk support
        '''
        self.uses_thunks = False  

class DAG:
    def __init__(self):
        self.graph = {}

    def add_entity(self, name, entity_type, dependencies=None):
        node = Node(name, entity_type, dependencies)
        self.graph[name] = node

    def get_dependencies(self, entity_name):
        '''
        Get entities that depend on the given entity, not its dependencies
        '''
        return [name for name, node in self.graph.items() if entity_name in node.dependencies]

    def get_result(self, entity_name):
        entity = self.graph[entity_name]
        if entity.dirty or any(self.graph[dep].dirty for dep in entity.dependencies):
            self._recompute(entity_name)
        '''
        Evaluate thunk if needed
        '''
        if entity.uses_thunks:  
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
        '''
        Since the "Total Compensation" value has no data to recompute, we can skip the recompute step
        '''
        if entity_name=="Total Compensation": 
            entity.dirty=False
            entity.dirty_rows.clear()
        dependencies = entity.dependencies 
        data = {dep: self.get_result(dep) for dep in dependencies}
        '''
        Loop for every dependency
        '''
        for dependency in dependencies:
            '''
            Check if any dependency is still dirty
            '''
            if self.graph[dependency].dirty:
                '''
                Recompute the dependency
                '''  
                self._recompute(dependency) 
                '''
                Create a handler based on the type of the entity
                '''
                if entity.type == "DataRowStore":   
                    handler = DataRowStore(data, entity_name)
                elif entity.type == "MathColumn":
                    handler = MathColumn(dependency.operation, dependency.input_columns, dependency)  
                ''' 
                Add more entity types as needed
                Set the result of the entity to be the computed output of the data using the defined handler
                '''
                entity.result = handler.recompute(data, self) 
        entity.dirty = False
        '''
        Propagate thunk usage
        '''
        if entity.uses_thunks:  
            for dependent_name in self.get_dependencies(entity_name):
                self.graph[dependent_name].uses_thunks = True

    def _topological_sort(self):
        ts = TopologicalSorter(self.graph)
        return list(ts.static_order())

    def _propagate_dirt(self, entity_name):
        entity = self.graph[entity_name]
        for dependent_name in self.get_dependencies(entity_name):
            dependent = self.graph[dependent_name]
            '''
            Update dirty rows of dependent
            '''
            dependent.dirty_rows.update(entity.dirty_rows)  
            dependent.dirty = True
            print(f"Propagating dirt to {dependent_name}")
        entity.dirty_rows.clear()
        print(f"Marked {entity_name} as dirty")
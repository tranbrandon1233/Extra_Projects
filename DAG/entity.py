
class Entity:
    def __init__(self, dependencies):
        self.dependencies = dependencies
        self.result = None
        self._dirty = True
        self.dirty_rows = set()  # Track dirty rows for each entity

    def is_dirty(self):
        return self._dirty

    def mark_clean(self):
        self._dirty = False

    def recompute(self, namespace):
        # Subclasses should implement their recomputation logic here
        # and update self.result
        pass
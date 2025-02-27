class BaseTask:
    def __init__(self, name):
        self.name = name
        self.prerequisites = []

class AdvancedTask(BaseTask):
    def __init__(self, name, difficulty):
        super().__init__(name)
        self.difficulty = difficulty


class TaskManager:
    def __init__(self, tasks):
        """
        tasks: List of BaseTask or AdvancedTask objects
        """
        self.tasks = tasks
        self.in_degree = {}
        self.adjacency = {}
        self.build_graph()

    def build_graph(self):
        for t in self.tasks:
            self.adjacency[t.name] = []
            self.in_degree[t.name] = 0

        for t in self.tasks:
            for prereq in t.prerequisites:
                self.adjacency[prereq.name].append(t.name)
                self.in_degree[t.name] += 1

    def find_task_order(self):
        from collections import deque

        # Queue for tasks with no prerequisites
        queue = deque([name for name in self.in_degree if self.in_degree[name] == 0])
        order = []

        while queue:
            current = queue.popleft()
            order.append(current)
            
            for neighbor in self.adjacency[current]:
                self.in_degree[neighbor] -= 1
                if self.in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(order) == len(self.tasks):
            return order
        else:
            return []
        
def test_valid_order():
    taskA = BaseTask("A")
    taskB = BaseTask("B")
    taskC = BaseTask("C")
    taskD = BaseTask("D")

    taskB.prerequisites = [taskA]
    taskC.prerequisites = [taskA]
    taskD.prerequisites = [taskB, taskC]

    tasks = [taskA, taskB, taskC, taskD]
    manager = TaskManager(tasks)
    order = manager.find_task_order()

    assert order in [["A", "B", "C", "D"], ["A", "C", "B", "D"]]

def test_no_prerequisites():
    tasks = [BaseTask("X"), BaseTask("Y"), BaseTask("Z")]
    manager = TaskManager(tasks)
    order = manager.find_task_order()
    assert set(order) == {"X", "Y", "Z"}

def test_single_task():
    tasks = [BaseTask("Single")]
    manager = TaskManager(tasks)
    order = manager.find_task_order()
    assert order == ["Single"]

def test_cycle_detection():
    task1 = BaseTask("1")
    task2 = BaseTask("2")
    task3 = BaseTask("3")

    task1.prerequisites = [task3]
    task2.prerequisites = [task1]
    task3.prerequisites = [task2]

    tasks = [task1, task2, task3]
    manager = TaskManager(tasks)
    order = manager.find_task_order()

    assert order == []

def test_advanced_tasks():
    adv_task1 = AdvancedTask("X", difficulty=3)
    adv_task2 = AdvancedTask("Y", difficulty=2)
    adv_task3 = AdvancedTask("Z", difficulty=1)

    adv_task2.prerequisites = [adv_task1]
    adv_task3.prerequisites = [adv_task2]

    tasks = [adv_task1, adv_task2, adv_task3]
    manager = TaskManager(tasks)
    order = manager.find_task_order()

    assert order == ["X", "Y", "Z"]

if __name__ == "__main__":
    test_valid_order()
    test_no_prerequisites()
    test_single_task()
    test_cycle_detection()
    test_advanced_tasks()
    print("All tests passed!")

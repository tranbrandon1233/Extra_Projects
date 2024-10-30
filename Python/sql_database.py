import sqlite3
from datetime import datetime, timedelta

# Database connection
conn = sqlite3.connect('task_manager.db')
cursor = conn.cursor()

# Create tables for projects and tasks
cursor.execute('''
    CREATE TABLE IF NOT EXISTS projects (
        id INTEGER PRIMARY KEY, 
        name TEXT
    )
''')

cursor.execute('''
    CREATE TABLE IF NOT EXISTS tasks (
        id INTEGER PRIMARY KEY, 
        title TEXT, 
        description TEXT, 
        priority TEXT, 
        due_date DATE, 
        completed INTEGER, 
        project_id INTEGER,
        recurring_interval INTEGER,
        FOREIGN KEY (project_id) REFERENCES projects(id)
    )
''')
conn.commit()

# Add a project
def create_project(name):
    cursor.execute("INSERT INTO projects (name) VALUES (?)", (name,))
    conn.commit()

# Delete a project (and associated tasks)
def delete_project(project_id):
    cursor.execute("DELETE FROM tasks WHERE project_id = ?", (project_id,))
    cursor.execute("DELETE FROM projects WHERE id = ?", (project_id,))
    conn.commit()

# Add a task
def add_task(title, description, priority, due_date, project_id, recurring_interval=None):
    cursor.execute('''
        INSERT INTO tasks (title, description, priority, due_date, completed, project_id, recurring_interval)
        VALUES (?, ?, ?, ?, 0, ?, ?)
    ''', (title, description, priority, due_date, project_id, recurring_interval))
    conn.commit()

# Update a task
def update_task(task_id, title=None, description=None, priority=None, due_date=None, completed=None, project_id=None):
    updates = []
    params = []
    
    if title:
        updates.append("title = ?")
        params.append(title)
    if description:
        updates.append("description = ?")
        params.append(description)
    if priority:
        updates.append("priority = ?")
        params.append(priority)
    if due_date:
        updates.append("due_date = ?")
        params.append(due_date)
    if completed is not None:
        updates.append("completed = ?")
        params.append(completed)
    if project_id:
        updates.append("project_id = ?")
        params.append(project_id)

    params.append(task_id)
    query = "UPDATE tasks SET " + ", ".join(updates) + " WHERE id = ?"
    cursor.execute(query, params)
    conn.commit()

# Delete a task
def delete_task(task_id):
    cursor.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
    conn.commit()

# Mark task as complete and handle recurring task logic
def complete_task(task_id):
    cursor.execute("SELECT recurring_interval, due_date, title, description, priority, project_id FROM tasks WHERE id = ?", (task_id,))
    result = cursor.fetchone()
    if result:
        interval, due_date, title, description, priority, project_id = result
        if interval:
            new_due_date = (datetime.strptime(due_date, '%Y-%m-%d') + timedelta(days=interval)).strftime('%Y-%m-%d')
            add_task(title, description, priority, new_due_date, project_id, interval)
        cursor.execute("UPDATE tasks SET completed = 1 WHERE id = ?", (task_id,))
        conn.commit()

# Get tasks with filtering and sorting options
def get_tasks(project_id=None, priority=None, completed=None, due_date_start=None, due_date_end=None, sort_by='due_date'):
    query = "SELECT * FROM tasks WHERE 1=1"
    params = []

    if project_id:
        query += " AND project_id = ?"
        params.append(project_id)
    if priority:
        query += " AND priority = ?"
        params.append(priority)
    if completed is not None:
        query += " AND completed = ?"
        params.append(completed)
    if due_date_start:
        query += " AND due_date >= ?"
        params.append(due_date_start)
    if due_date_end:
        query += " AND due_date <= ?"
        params.append(due_date_end)

    query += f" ORDER BY {sort_by}"
    cursor.execute(query, params)
    return cursor.fetchall()

# Unit Testing
import unittest

class TestTaskManager(unittest.TestCase):
    def setUp(self):
        # Create test project and tasks
        cursor.execute("DELETE FROM tasks")
        cursor.execute("DELETE FROM projects")
        create_project("Test Project")
        project_id = cursor.lastrowid
        add_task("Test Task 1", "Test Description 1", "High", "2024-10-22", project_id)
        add_task("Test Task 2", "Test Description 2", "Low", "2024-10-30", project_id, 7)

    def tearDown(self):
        # Cleanup after tests
        cursor.execute("DELETE FROM tasks")
        cursor.execute("DELETE FROM projects")
        conn.commit()

    def test_add_task(self):
        add_task("New Task", "New Description", "Medium", "2024-11-01", 1)
        tasks = get_tasks()
        self.assertTrue(any(task[1] == "New Task" for task in tasks))

    def test_update_task(self):
        tasks = get_tasks()
        task_id = tasks[0][0]
        update_task(task_id, title="Updated Task")
        updated_task = get_tasks()[0]
        self.assertEqual(updated_task[1], "Updated Task")

    def test_delete_task(self):
        tasks = get_tasks()
        task_id = tasks[0][0]
        delete_task(task_id)
        remaining_tasks = get_tasks()
        self.assertFalse(any(task[0] == task_id for task in remaining_tasks))

    def test_complete_task(self):
        tasks = get_tasks()
        task_id = tasks[1][0]  # Task with recurring interval
        complete_task(task_id)
        new_tasks = get_tasks()
        self.assertTrue(any(task[1] == "Test Task 2" and task[5] == 0 for task in new_tasks))  # Check for new, incomplete task

    def test_filter_tasks(self):
        tasks = get_tasks(priority="High")
        self.assertTrue(all(task[3] == "High" for task in tasks))

    def test_sort_tasks(self):
        tasks = get_tasks(sort_by="priority")
        self.assertEqual(tasks[0][3], "High")
        self.assertEqual(tasks[1][3], "Low")

    def test_project_management(self):
        create_project("New Project")
        project_id = cursor.lastrowid
        add_task("Project Task", "Desc", "Medium", "2024-11-01", project_id)
        tasks = get_tasks(project_id=project_id)
        self.assertTrue(any(task[1] == "Project Task" for task in tasks))

if __name__ == '__main__':
    unittest.main()
    
# Close the connection
conn.close()
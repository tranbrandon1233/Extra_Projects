from flask import Flask, render_template, request, jsonify, abort
app = Flask(__name__)

# Sample data for our to-do list
todos = [
    {
        'id': 1,
        'title': 'Buy groceries',
        'completed': False
    },
    {
        'id': 2,
        'title': 'Learn Flask',
        'completed': True
    }
]

@app.route('/', defaults={'todo_id': None}, methods=['GET'])
@app.route('/<int:todo_id>', methods=['GET'])
def get_todos(todo_id):
    # Handle GET request to fetch a todo by the specified ID if provided
    if todo_id:
        # Fetch a specific todo by ID
        for todo in todos:
            if todo['id'] == todo_id:
                return jsonify(todo)
        # Return error if todo not found
        abort(404,description='Todo not found')
    else:
        # Return all todos
        return jsonify({'todos': todos})

@app.route('/', methods=['POST'])
def add_todo():
    # Handle POST request to add a new todo
    data = request.get_json()
    # Error handling if received data is not valid
    if not data or 'title' not in data:
        abort(400, description="Invalid data")
    todo = {
        'id': len(todos) + 1,
        'title': data['title'],
        'completed': False
    }
    # Append the new todo to the list
    todos.append(todo)
    return jsonify({'message': 'Todo added successfully', 'todo': todo}), 201

@app.route('/<int:todo_id>', methods=['PUT'])
def update_todo(todo_id):
    # Handle PUT request to update existing todo
    data = request.get_json()
    # Error handling to ensure data received is valid
    if not data or 'title' not in data or 'completed' not in data:
        return abort(400,description="Invalid data")
    # Return todo based on ID
    for todo in todos:
        if todo['id'] == todo_id:
            todo['title'] = data['title']
            todo['completed'] = data['completed']
            return jsonify({'message': 'Todo updated successfully', 'todo': todo})
    abort(404,description='Todo not found')

@app.route('/<int:todo_id>', methods=['DELETE'])
def delete_todo(todo_id):
    # Handle DELETE request to delete a todo
    global todos
    # Delete the todo based on ID
    for i, todo in enumerate(todos):
        if todo['id'] == todo_id:
            todos.pop(i)
            return jsonify({'message': 'Todo deleted successfully'})
    # Return error if todo not found
    abort(404,description='Todo not found')

if __name__ == '__main__':
    # Run the flask app
    app.run(debug=True)  
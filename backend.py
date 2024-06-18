from flask import Flask, jsonify, request
from flask_cors import CORS
from dag import DAG, DataRowStore, MathColumn, Entity
from entity_handlers import DataRowStore

app = Flask(__name__)
CORS(app)

# Initialize the DAG
namespace = DAG()

# Add sample data row store entities
salary_data = {"Salary": [100, 120, 80]}
bonus_data = {"Bonus": [20, 30, 10]}
namespace.add_entity("Salary", "DataRowStore")
namespace.add_entity("Bonus", "DataRowStore")

# Add a sample computed column entity
namespace.add_entity("Total Compensation", "MathColumn", ["Salary", "Bonus"])
namespace.graph["Total Compensation"].operation = "add"
namespace.graph["Total Compensation"].input_columns = ["Salary.Salary", "Bonus.Bonus"]

# Load initial data into data row stores
namespace.graph["Salary"].result = DataRowStore(salary_data, "Salary").recompute()
namespace.graph["Bonus"].result = DataRowStore(bonus_data, "Bonus").recompute()

@app.route('/api/dag')
def get_dag():
    return jsonify(dag_data)

@app.route('/api/data-row-store')
def get_data_row_store():
    return jsonify(data_row_store_data)

@app.route('/api/computed-columns')
def get_computed_columns():
    return jsonify(computed_columns_data)

@app.route('/api/tasks', methods=['GET', 'POST'])
def handle_tasks():
    if request.method == 'POST':
        new_task = request.json
        tasks_data.append(new_task)
        return jsonify(new_task), 201
    else:
        return jsonify(tasks_data)

@app.route('/api/tasks/<int:task_index>', methods=['PUT'])
def update_task(task_index):
    if task_index < len(tasks_data):
        task_update = request.json
        tasks_data[task_index] = task_update
        return jsonify(task_update)
    else:
        return "Task not found", 404

@app.route('/api/update-data-row-store', methods=['POST'])
def update_data_row_store():
    updated_values = request.json
    for entity_name, data in updated_values.items():
        namespace.graph[entity_name].result = DataRowStore(data, entity_name).recompute(data, namespace)

    # Return updated DAG and computed columns data
    return jsonify({
        "dag_data": {
            "dot": namespace.generate_dot()
        },
        "data_row_store_data": {
            entity_name: entity.result for entity_name, entity in namespace.graph.items()
            if entity.type == "DataRowStore"
        },
        "computed_columns_data": {
            entity_name: entity.result for entity_name, entity in namespace.graph.items()
            if entity.type == "MathColumn"
        }
    })

# Make sure to enable CORS for all routes
CORS(app, resources={r"/api/*": {"origins": "*"}})

if __name__ == '__main__':
    app.run(port=5000, debug=True)
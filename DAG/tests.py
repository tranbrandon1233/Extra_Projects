
import unittest
from entity_handlers import DataRowStore, MathColumn  # Import from entity_handlers.py
from dag import DAG

class TestDirtPropagation(unittest.TestCase):
    def test_data_row_store_dirt(self):
        namespace = DAG()
        salary = DataRowStore({"Salary": 100}, "Salary")
        bonus = DataRowStore({"Bonus": 20}, "Bonus")
        total_comp = MathColumn("add", ["Salary.Salary", "Bonus.Bonus"], "Total Compensation")
        namespace.add_entity("Salary", "DataRowStore")
        namespace.add_entity("Bonus", "DataRowStore")
        namespace.add_entity("Total Compensation", "MathColumn", ["Salary", "Bonus"])

        # Initial state
        self.assertFalse(namespace.graph['Total Compensation'].dirty)
        self.assertEqual(namespace.graph['Total Compensation'].dirty_rows, set())

        # Update data in DataRowStore
        salary.recompute({"Salary": 120}, namespace)

        # Check if Total Compensation is dirty for the updated row
        self.assertTrue(namespace.graph["Total Compensation"].dirty)
        self.assertEqual(namespace.graph["Total Compensation"].dirty_rows, {0})  # Assuming row index 0 was updated

        # Update data in another DataRowStore
        bonus.recompute({"Bonus": 30}, namespace)

        # Check if Total Compensation is dirty for both updated rows
        self.assertTrue(namespace.graph["Total Compensation"].dirty)
        self.assertEqual(namespace.graph["Total Compensation"].dirty_rows, {0, 0})  # Both rows with index 0
        # Recompute Total Compensation
        namespace.get_result("Total Compensation") 

        # Check if Total Compensation is clean
        self.assertFalse(namespace.graph["Total Compensation"].dirty)
        self.assertEqual(namespace.graph["Total Compensation"].dirty_rows, set())

class TestEntityHandlers(unittest.TestCase):
    def test_data_row_store(self):
        data = {"Salary": 100, "Bonus": 20}
        entity = DataRowStore(data, "TestDataRowStore")
        result = entity.recompute()
        self.assertEqual(result, data)

    def test_math_column_add(self):
        data = {
            "Salary": {"Salary": [100]},  # Nested dictionary with list for Salary entity
            "Bonus": {"Bonus": [20]}     # Nested dictionary with list for Bonus entity
        }
        entity = MathColumn("add", ["Salary.Salary", "Bonus.Bonus"], "TestMathColumn")
        result = entity.recompute(data)
        self.assertEqual(result, {0: 120})  # Assuming r

if __name__ == "__main__":
    unittest.main()
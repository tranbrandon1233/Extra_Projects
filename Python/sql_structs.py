import unittest
from pyspark.sql.types import (
    ArrayType,
    FloatType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

txo_schema = StructType(
    [
        StructField("id", StringType(), True),
        StructField("address", StringType(), True),
        StructField("value", IntegerType(), True),
    ]
)

visit_txo_schema = StructType(
    [
        StructField("txo", txo_schema, True),
        StructField("share", FloatType(), True),
        StructField("hop", IntegerType(), True),
    ]
)

sof_dof_schema = StructType(
    [
        StructField("amount", IntegerType(), True),
        StructField("label", StringType(), True),
        StructField("category", StringType(), True),
        StructField("detail", ArrayType(visit_txo_schema), True),
    ]
)


class TestSchemas(unittest.TestCase):
    def test_txo_schema(self):
        expected_fields = ["id", "address", "value"]
        expected_types = [StringType(), StringType(), IntegerType()]
        self.assertEqual(len(txo_schema), 3)
        for i, field in enumerate(txo_schema):
            self.assertEqual(field.name, expected_fields[i])
            self.assertEqual(type(field.dataType), type(expected_types[i]))

    def test_visit_txo_schema(self):
        expected_fields = ["txo", "share", "hop"]
        expected_types = [txo_schema, FloatType(), IntegerType()]
        self.assertEqual(len(visit_txo_schema), 3)
        for i, field in enumerate(visit_txo_schema):
            self.assertEqual(field.name, expected_fields[i])
            self.assertEqual(type(field.dataType), type(expected_types[i]))

    def test_sof_dof_schema(self):
        expected_fields = ["amount", "label", "category", "detail"]
        expected_types = [
            IntegerType(),
            StringType(),
            StringType(),
            ArrayType(visit_txo_schema),
        ]
        self.assertEqual(len(sof_dof_schema), 4)
        for i, field in enumerate(sof_dof_schema):
            self.assertEqual(field.name, expected_fields[i])
            self.assertEqual(type(field.dataType), type(expected_types[i]))


if __name__ == "__main__":
    unittest.main()
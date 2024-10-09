import asyncio
import pandas as pd
import pytest
from fhir.resources import Coding, Condition
from pydantic import BaseModel, ValidationError
from emr.epic.models.basic_epic_etl import BasicEpicEtl
from emr.shared.fhir.types.coding_system_uri import CodingSystemUri

from your_script import Problems  # Replace 'your_script' with the actual script's name

# Sample data
sample_coding_list = [
    Coding(system=CodingSystemUri.SNOMED_CT, code="271349000", display="Type 1 diabetes"),
    Coding(system=CodingSystemUri.ICD_10_CM, code="E11.9", display="Type 1 diabetes mellitus, unspecified"),
    Coding(system=CodingSystemUri.LOINC, code="205294004", display="Diabetes mellitus, type 1"),
]


class TestProblems:
    def test_relevant_columns(self):
        assert Problems.relevant_columns == [
            "patient_id",
            "encounter_id",
            "problem_identifier",
            "superceded_indication.display",
            "problem_reason.code",
            "problem_status",
            "problem_onset_date",
            "problem_abatement_date",
            "entry_encounter_id",
            "source_id",
        ]

    def test_table_name(self):
        assert Problems.table_name == "problems"

    @pytest.mark.asyncio
    async def test_extract_empty_result(self, mocker):
        mocker.patch("emr.epic.models.basic_epic_etl.BasicEpicEtl.extract", return_value=[])
        problems = Problems()
        result = await problems.extract()
        assert result == []
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_extract_valid_result(self, mocker):
        mock_condition_resource = Condition(
            id="condition-123",
            status="active",
            code=sample_coding_list,
            onsetDateTime="2023-10-26T10:00:00Z",
        )
        mocker.patch(
            "emr.epic.models.basic_epic_etl.BasicEpicEtl.extract",
            return_value=[mock_condition_resource],
        )
        problems = Problems()
        result = await problems.extract()
        assert len(result) == 1
        condition = result[0]
        assert isinstance(condition, Condition)
        assert condition.id == "condition-123"
        assert condition.status == "active"
        assert condition.code == sample_coding_list
        assert condition.onsetDateTime == "2023-10-26T10:00:00Z"

    def test_transform_empty_list(self):
        problems = Problems()
        result = problems.transform([])
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_transform_valid_list(self):
        condition_data = [
            {
                "patient_id": "12345",
                "encounter_id": "67890",
                "problem_identifier": "condition-123",
                "superceded_indication.display": "Initial onset diabetes symptoms",
                "problem_reason.code": "271349000",
                "problem_status": "active",
                "problem_onset_date": "2023-10-26",
                "problem_abatement_date": None,
                "entry_encounter_id": "67890",
                "source_id": "source-1",
            },
            {
                "patient_id": "12345",
                "encounter_id": "67890",
                "problem_identifier": "condition-456",
                "superceded_indication.display": "Hypertension follow-up",
                "problem_reason.code": "401.1",
                "problem_status": "inactive",
                "problem_onset_date": "2022-05-15",
                "problem_abatement_date": "2023-06-30",
                "entry_encounter_id": "123456",
                "source_id": "source-2",
            },
        ]
        problems = Problems()
        result = problems.transform(condition_data)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert result.columns.tolist() == Problems.relevant_columns

        # Check specific row values
        expected_first_row = {
            "patient_id": "12345",
            "encounter_id": "67890",
            "problem_identifier": "condition-123",
            "superceded_indication.display": "Initial onset diabetes symptoms",
            "problem_reason.code": "271349000",
            "problem_status": "active",
            "problem_onset_date": pd.Timestamp("2023-10-26"),
            "problem_abatement_date": pd.NaT,
            "entry_encounter_id": "67890",
            "source_id": "source-1",
        }
        pd.testing.assert_series_equal(result.iloc[0], pd.Series(expected_first_row))

    def test_get_codes_from_coding_list(self):
        result = Problems._get_codes_from_coding_list(sample_coding_list)
        expected_result = {
            CodingSystemUri.SNOMED_CT: ["271349000"],
            CodingSystemUri.ICD_10_CM: ["E11.9"],
            CodingSystemUri.LOINC: ["205294004"],
        }
        assert result == expected_result

    def test_get_codes_from_coding_list_empty(self):
        result = Problems._get_codes_from_coding_list([])
        assert result == {}

    def test_get_codes_from_coding_list_duplicates(self):
        coding_list_with_duplicates = sample_coding_list + [
            Coding(system=CodingSystemUri.SNOMED_CT, code="271349000", display="Type 1 diabetes")
        ]
        result = Problems._get_codes_from_coding_list(coding_list_with_duplicates)
        # Duplicates should be removed in the dictionary
        expected_result = {
            CodingSystemUri.SNOMED_CT: ["271349000"],
            CodingSystemUri.ICD_10_CM: ["E11.9"],
            CodingSystemUri.LOINC: ["205294004"],
        }
        assert result == expected_result

    @pytest.mark.asyncio
    async def test_extract_invalid_data(self, mocker):
        mocker.patch(
            "emr.epic.models.basic_epic_etl.BasicEpicEtl.extract",
            return_value=[{"invalid": "data"}],
        )
        problems = Problems()
        with pytest.raises(ValidationError):
            await problems.extract()

    # (Add more specific tests if the script has more complex logic)
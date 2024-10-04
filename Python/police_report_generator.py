import datetime
import openpyxl

class SovereignCitizenReport:
    """Generates standardized reports for sovereign citizen encounters."""

    def __init__(self, officer_name, date, time, location, plate_number, vehicle_description):
        """
        Initializes the report with basic encounter information.

        Args:
            officer_name: The name of the reporting officer.
            date: The date of the encounter (YYYY-MM-DD).
            time: The time of the encounter (HH:MM:SS - 24-hour format).
            location: The encounter location.
            plate_number: The vehicle's license plate number.
            vehicle_description: A description of the vehicle.
        """
        self.officer_name = officer_name
        self.date = date
        self.time = time
        self.location = location
        self.plate_number = plate_number
        self.vehicle_description = vehicle_description
        
        # Initialize report sections
        self.identification_details = {}
        self.statements = []
        self.actions_taken = []
        self.additional_notes = []

    def add_identification_details(self, name, age, dob, address, driver_license, other_ids):
        """
        Records individual identification details.

        Args:
            name: Full name of the individual.
            age: Individual's age.
            dob: Date of birth.
            address: Individual's address.
            driver_license: Driver's license number (or "None").
            other_ids: Any other identification presented (list of strings).
        """
        self.identification_details = {
            "name": name,
            "age": age,
            "dob": dob,
            "address": address,
            "driver_license": driver_license or "None",
            "other_ids": other_ids
        }

    def add_statement(self, speaker, text):
        """
        Records a statement made during the encounter.

        Args:
            speaker: Who made the statement (e.g., "officer", "subject").
            text: The full text of the statement.
        """
        self.statements.append({"speaker": speaker, "text": text})

    def add_action_taken(self, action, description):
        """
        Records an action taken by the officer during the encounter.

        Args:
            action: A concise description of the action.
            description: A more detailed explanation of the action.
        """
        self.actions_taken.append({"action": action, "description": description})

    def add_note(self, note):
        """
        Adds an additional note or observation about the encounter.

        Args:
            note: The note to be added.
        """
        self.additional_notes.append(note)
    
    def generate_excel_report(self, filename="report.xlsx"):
        """
        Generates a detailed Excel report of the encounter.
        """
        workbook = openpyxl.Workbook()
        sheet = workbook.active

        # Add report header
        sheet['A1'] = "Sovereign Citizen Encounter Report"
        sheet['A2'] = f"Date: {self.date}    Time: {self.time}"
        sheet['A3'] = f"Officer: {self.officer_name}"
        sheet['A4'] = f"Location: {self.location}"
        sheet['A5'] = f"Plate Number: {self.plate_number}"
        sheet['A6'] = f"Vehicle Description: {self.vehicle_description}"
        sheet.merge_cells('A1:I1')
        sheet.merge_cells('A2:I2')

        # Identification Details section
        sheet['A7'] = "Identification Details"
        sheet.merge_cells('A7:B7')
        row = 8
        for key, value in self.identification_details.items():
            sheet[f'A{row}'] = key.capitalize()
            sheet[f'B{row}'] = value
            row += 1

        # Statements section
        sheet['A' + str(row)] = "Statements"
        sheet.merge_cells(f'A{row}:B{row}')
        row += 1
        sheet['A' + str(row)] = "Speaker"
        sheet['B' + str(row)] = "Text"
        row += 1
        for statement in self.statements:
            sheet[f'A{row}'] = statement["speaker"]
            sheet[f'B{row}'] = statement["text"]
            row += 1

        # Actions Taken section
        sheet['A' + str(row)] = "Actions Taken"
        sheet.merge_cells(f'A{row}:B{row}')
        row += 1
        sheet['A' + str(row)] = "Action"
        sheet['B' + str(row)] = "Description"
        row += 1
        for action in self.actions_taken:
            sheet[f'A{row}'] = action["action"]
            sheet[f'B{row}'] = action["description"]
            row += 1

        # Additional Notes section
        if self.additional_notes:
            sheet['A' + str(row)] = "Additional Notes"
            sheet.merge_cells(f'A{row}:B{row}')
            row += 1
            for note in self.additional_notes:
                sheet[f'A{row}'] = note
                row += 1

        workbook.save(filename)
        print(f"Report saved as {filename}.")
    
    def generate_text_report(self, filename="report.txt"):
        """
        Generates a plain text report of the encounter.
        """
        with open(filename, "w", encoding="utf-8") as f:
            # Report header
            f.write("Sovereign Citizen Encounter Report\n")
            f.write(f"Date: {self.date}   Time: {self.time}\n")
            f.write(f"Officer: {self.officer_name}\n")
            f.write(f"Location: {self.location}\n")
            f.write(f"Plate Number: {self.plate_number}\n")
            f.write(f"Vehicle Description: {self.vehicle_description}\n\n")

            # Identification Details
            f.write("Identification Details:\n")
            for key, value in self.identification_details.items():
                f.write(f"{key.capitalize()}: {value}\n")
            f.write("\n")

            # Statements
            f.write("Statements:\n")
            for statement in self.statements:
                f.write(f"Speaker: {statement['speaker']}\n")
                f.write(f"Text: {statement['text']}\n\n")

            # Actions Taken
            f.write("Actions Taken:\n")
            for action in self.actions_taken:
                f.write(f"Action: {action['action']}\n")
                f.write(f"Description: {action['description']}\n\n")

            # Additional Notes
            if self.additional_notes:
                f.write("Additional Notes:\n")
                for note in self.additional_notes:
                    f.write(f"{note}\n")

        print(f"Report saved as {filename}.")


# Example Usage:
def main():
    """
    Demonstrates how to use the SovereignCitizenReport class.
    """
    now = datetime.datetime.now()
    report = SovereignCitizenReport(
        officer_name="John Doe",
        date=now.strftime("%Y-%m-%d"),
        time=now.strftime("%H:%M:%S"),
        location="Main Street, Anytown",
        plate_number="123ABC",
        vehicle_description="Blue Sedan",
    )

    report.add_identification_details(
        name="Jane Citizen",
        age=45,
        dob="1978-06-15",
        address="123 Main Street, Anytown",
        driver_license="123456789",
        other_ids=["State ID Card"],
    )

    report.add_statement(speaker="officer", text="Requested to see driver's license and registration.")
    report.add_statement(speaker="subject", text="Declined to provide identification, claiming I'm not subject to traffic laws.")
    report.add_statement(speaker="officer", text="Informed subject of the requirements for driving a vehicle.")
    report.add_statement(speaker="subject", text="Refused to comply and began reciting legal jargon.")
    report.add_statement(speaker="officer", text="Issued a traffic citation for driving without a valid license")
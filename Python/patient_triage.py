import heapq
from datetime import datetime, timedelta
import threading
import time

class Doctor:
    def __init__(self, id, name, specialty, experience_level, schedule, max_patients, clinic_schedule=None):
        self.id = id
        self.name = name
        self.specialty = specialty
        self.experience_level = experience_level
        self.schedule = schedule  # List of (start_time, end_time) tuples
        self.max_patients = max_patients  # Maximum number of patients a doctor can handle
        self.current_patients = []
        self.clinic_schedule = clinic_schedule  # Optional schedule for specialized clinics

    def is_available(self, time):
        for start, end in self.schedule:
            if start <= time <= end:
                return True
        return False

    def is_in_clinic(self, time):
        if self.clinic_schedule:
            for start, end in self.clinic_schedule:
                if start <= time <= end:
                    return True
        return False

    def update_availability(self):
        # Remove patients who are done with treatment to free up the doctor's workload
        self.current_patients = [p for p in self.current_patients if not p.is_treatment_done()]

class Patient:
    def __init__(self, id, name, department, severity, treatment_duration, clinic_required=False):
        self.id = id
        self.name = name
        self.department = department
        self.severity = severity
        self.wait_start_time = datetime.now()
        self.treatment_end_time = None
        self.treatment_duration = treatment_duration  # Duration in minutes
        self.clinic_required = clinic_required  # Whether the patient requires a specialized clinic

    def start_treatment(self):
        self.treatment_end_time = datetime.now() + timedelta(minutes=self.treatment_duration)

    def is_treatment_done(self):
        if self.treatment_end_time and datetime.now() >= self.treatment_end_time:
            return True
        return False

# Data Structures
doctors = []  # List of Doctor objects
patients_queue = []  # Priority queue of (severity, patient) tuples
waiting_list = []  # List of patients waiting for a doctor
notifications = []  # List of notifications for doctors and patients

# Assign doctor to patient
def assign_doctor(patient):
    # Update doctor availability
    for doctor in doctors:
        doctor.update_availability()

    # Filter available doctors based on department, schedule, and workload
    available_doctors = [d for d in doctors if d.specialty == patient.department and 
                         d.is_available(datetime.now()) and 
                         len(d.current_patients) < d.max_patients]

    # If the patient requires a specialized clinic, filter doctors by clinic schedule
    if patient.clinic_required:
        available_doctors = [d for d in available_doctors if d.is_in_clinic(datetime.now())]

    if not available_doctors:
        waiting_list.append(patient)
        notifications.append(f"Patient {patient.name} added to waiting list.")
        return None

    # Select the best doctor based on experience level and current workload
    best_doctor = min(available_doctors, key=lambda d: (-d.experience_level, len(d.current_patients)))
    best_doctor.current_patients.append(patient)
    patient.start_treatment()
    notifications.append(f"Patient {patient.name} assigned to Dr. {best_doctor.name}")
    return best_doctor

# Handle new patient arrivals
def handle_new_patient(patient):
    if patient.severity >= 8:  # Emergency
        doctor = assign_doctor(patient)
        if doctor:
            print(f"Emergency patient {patient.name} assigned to Dr. {doctor.name}")
        else:
            print(f"Emergency patient {patient.name} added to waiting list.")
    else:
        heapq.heappush(patients_queue, (-patient.severity, patient, datetime.now()))

# Process the patient queue dynamically
def process_queue():
    while True:
        if patients_queue:
            _, patient, arrival_time = heapq.heappop(patients_queue)
            doctor = assign_doctor(patient)
            if doctor:
                print(f"Patient {patient.name} assigned to Dr. {doctor.name}")
            else:
                print(f"Patient {patient.name} added to waiting list.")
                heapq.heappush(patients_queue, (-patient.severity, patient, arrival_time))
        time.sleep(1)  # Check the queue every second

# Reassign patients from the waiting list as doctors become available
def reassign_waiting_list():
    while True:
        updated_waiting_list = []
        # Sort waiting list based on severity and waiting time
        sorted_waiting_list = sorted(waiting_list, key=lambda p: (-p.severity, p.wait_start_time))
        for patient in sorted_waiting_list:
            doctor = assign_doctor(patient)
            if doctor:
                print(f"Patient {patient.name} from waiting list assigned to Dr. {doctor.name}")
            else:
                updated_waiting_list.append(patient)
        waiting_list[:] = updated_waiting_list
        time.sleep(5)  # Reassign patients from waiting list every 5 seconds

# Notification system to print notifications
def notification_system():
    while True:
        if notifications:
            notification = notifications.pop(0)
            print(notification)
        time.sleep(1)  # Check for new notifications every second


# Create sample doctors with different specialties, schedules, and experience levels
doctors = [
    Doctor(
        id=1,
        name="Sarah Chen",
        specialty="Cardiology",
        experience_level=9,
        schedule=[(datetime.now().replace(hour=8, minute=0), datetime.now().replace(hour=17, minute=0))],
        max_patients=5,
        clinic_schedule=[(datetime.now().replace(hour=14, minute=0), datetime.now().replace(hour=17, minute=0))]
    ),
    Doctor(
        id=2,
        name="Michael Rodriguez",
        specialty="Emergency",
        experience_level=8,
        schedule=[(datetime.now().replace(hour=7, minute=0), datetime.now().replace(hour=19, minute=0))],
        max_patients=8
    ),
    Doctor(
        id=3,
        name="Emily Johnson",
        specialty="Pediatrics",
        experience_level=7,
        schedule=[(datetime.now().replace(hour=9, minute=0), datetime.now().replace(hour=16, minute=0))],
        max_patients=6,
        clinic_schedule=[(datetime.now().replace(hour=9, minute=0), datetime.now().replace(hour=12, minute=0))]
    ),
    Doctor(
        id=4,
        name="James Wilson",
        specialty="Cardiology",
        experience_level=6,
        schedule=[(datetime.now().replace(hour=12, minute=0), datetime.now().replace(hour=20, minute=0))],
        max_patients=4
    ),
    Doctor(
        id=5,
        name="Lisa Thompson",
        specialty="Emergency",
        experience_level=9,
        schedule=[(datetime.now().replace(hour=19, minute=0), datetime.now().replace(hour=7, minute=0))],
        max_patients=8
    ),
    Doctor(
        id=6,
        name="Robert Brown",
        specialty="Pediatrics",
        experience_level=8,
        schedule=[(datetime.now().replace(hour=8, minute=0), datetime.now().replace(hour=16, minute=0))],
        max_patients=6
    )
]

# Create sample patients with varying severity levels and requirements
sample_patients = [
    Patient(
        id=1,
        name="John Smith",
        department="Cardiology",
        severity=9,
        treatment_duration=60,
        clinic_required=True
    ),
    Patient(
        id=2,
        name="Maria Garcia",
        department="Emergency",
        severity=10,
        treatment_duration=30,
        clinic_required=False
    ),
    Patient(
        id=3,
        name="David Lee",
        department="Cardiology",
        severity=5,
        treatment_duration=45,
        clinic_required=False
    ),
    Patient(
        id=4,
        name="Sarah Brown",
        department="Pediatrics",
        severity=7,
        treatment_duration=40,
        clinic_required=False
    ),
    Patient(
        id=5,
        name="Robert Taylor",
        department="Emergency",
        severity=8,
        treatment_duration=20,
        clinic_required=False
    ),
    Patient(
        id=6,
        name="Emma Davis",
        department="Pediatrics",
        severity=4,
        treatment_duration=30,
        clinic_required=False
    ),
    Patient(
        id=7,
        name="Michael Wilson",
        department="Cardiology",
        severity=6,
        treatment_duration=50,
        clinic_required=True
    ),
    Patient(
        id=8,
        name="Jessica Martinez",
        department="Emergency",
        severity=9,
        treatment_duration=25,
        clinic_required=False
    )
]

# Create sample doctors with different specialties, schedules, and experience levels
doctors = [
    Doctor(
        id=1,
        name="Sarah Chen",
        specialty="Cardiology",
        experience_level=9,
        schedule=[(datetime.now().replace(hour=8, minute=0), datetime.now().replace(hour=17, minute=0))],
        max_patients=5,
        clinic_schedule=[(datetime.now().replace(hour=14, minute=0), datetime.now().replace(hour=17, minute=0))]
    ),
    Doctor(
        id=2,
        name="Michael Rodriguez",
        specialty="Emergency",
        experience_level=8,
        schedule=[(datetime.now().replace(hour=7, minute=0), datetime.now().replace(hour=19, minute=0))],
        max_patients=8
    ),
    Doctor(
        id=3,
        name="Emily Johnson",
        specialty="Pediatrics",
        experience_level=7,
        schedule=[(datetime.now().replace(hour=9, minute=0), datetime.now().replace(hour=16, minute=0))],
        max_patients=6,
        clinic_schedule=[(datetime.now().replace(hour=9, minute=0), datetime.now().replace(hour=12, minute=0))]
    ),
    Doctor(
        id=4,
        name="James Wilson",
        specialty="Cardiology",
        experience_level=6,
        schedule=[(datetime.now().replace(hour=12, minute=0), datetime.now().replace(hour=20, minute=0))],
        max_patients=4
    ),
    Doctor(
        id=5,
        name="Lisa Thompson",
        specialty="Emergency",
        experience_level=9,
        schedule=[(datetime.now().replace(hour=19, minute=0), datetime.now().replace(hour=7, minute=0))],
        max_patients=8
    ),
    Doctor(
        id=6,
        name="Robert Brown",
        specialty="Pediatrics",
        experience_level=8,
        schedule=[(datetime.now().replace(hour=8, minute=0), datetime.now().replace(hour=16, minute=0))],
        max_patients=6
    )
]

# Create sample patients with varying severity levels and requirements
sample_patients = [
    Patient(
        id=1,
        name="John Smith",
        department="Cardiology",
        severity=9,
        treatment_duration=60,
        clinic_required=True
    ),
    Patient(
        id=2,
        name="Maria Garcia",
        department="Emergency",
        severity=10,
        treatment_duration=30,
        clinic_required=False
    ),
    Patient(
        id=3,
        name="David Lee",
        department="Cardiology",
        severity=5,
        treatment_duration=45,
        clinic_required=False
    ),
    Patient(
        id=4,
        name="Sarah Brown",
        department="Pediatrics",
        severity=7,
        treatment_duration=40,
        clinic_required=False
    ),
    Patient(
        id=5,
        name="Robert Taylor",
        department="Emergency",
        severity=8,
        treatment_duration=20,
        clinic_required=False
    ),
    Patient(
        id=6,
        name="Emma Davis",
        department="Pediatrics",
        severity=4,
        treatment_duration=30,
        clinic_required=False
    ),
    Patient(
        id=7,
        name="Michael Wilson",
        department="Cardiology",
        severity=6,
        treatment_duration=50,
        clinic_required=True
    ),
    Patient(
        id=8,
        name="Jessica Martinez",
        department="Emergency",
        severity=9,
        treatment_duration=25,
        clinic_required=False
    )
]

# Modified handle_new_patient function to avoid duplicate printing
def handle_new_patient(patient):
    if patient.severity >= 8:  # Emergency
        doctor = assign_doctor(patient)
        if not doctor:
            notifications.append(f"Emergency patient {patient.name} added to waiting list.")
    else:
        heapq.heappush(patients_queue, (-patient.severity, patient, datetime.now()))

# Modified process_queue function to avoid duplicate printing
def process_queue():
    while True:
        if patients_queue:
            _, patient, arrival_time = heapq.heappop(patients_queue)
            doctor = assign_doctor(patient)
            if not doctor:
                heapq.heappush(patients_queue, (-patient.severity, patient, arrival_time))
        time.sleep(1)

# Modified reassign_waiting_list function to avoid duplicate printing
def reassign_waiting_list():
    while True:
        updated_waiting_list = []
        sorted_waiting_list = sorted(waiting_list, key=lambda p: (-p.severity, p.wait_start_time))
        for patient in sorted_waiting_list:
            doctor = assign_doctor(patient)
            if not doctor:
                updated_waiting_list.append(patient)
        waiting_list[:] = updated_waiting_list
        time.sleep(5)

# Function to simulate patient arrivals
def simulate_patient_arrivals():
    for patient in sample_patients:
        print(f"\nProcessing patient: {patient.name} (Severity: {patient.severity}, Department: {patient.department})")
        handle_new_patient(patient)
        time.sleep(2)  # Simulate time between patient arrivals

if __name__ == "__main__":
    # Print available doctors and their specialties
    print("Available Doctors:")
    for doctor in doctors:
        print(f"Dr. {doctor.name} - {doctor.specialty}")
    print("\nStarting patient processing...\n")

    # Start the system threads
    queue_thread = threading.Thread(target=process_queue, daemon=True)
    waiting_list_thread = threading.Thread(target=reassign_waiting_list, daemon=True)
    notification_thread = threading.Thread(target=notification_system, daemon=True)

    queue_thread.start()
    waiting_list_thread.start()
    notification_thread.start()

    # Start patient simulation
    simulate_patient_arrivals()

    # Keep the main program running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down the system...")
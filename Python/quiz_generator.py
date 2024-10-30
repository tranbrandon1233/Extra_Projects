import random
import time

# Data storage placeholders (replace with a database in a real application)
students = {}
admins = {"admin": "password"}  # Admin credentials (username: password)
questions = [
    {"question": "What is 2+2?", "options": ["3", "4", "5"], "answer": "4"},
    {"question": "What is the capital of France?", "options": ["Paris", "London", "Berlin"], "answer": "Paris"},
    {"question": "What is 3*3?", "options": ["6", "9", "12"], "answer": "9"},
    {"question": "What is the largest ocean?", "options": ["Atlantic", "Indian", "Pacific"], "answer": "Pacific"},
    {"question": "What is the chemical symbol for water?", "options": ["H2O", "O2", "CO2"], "answer": "H2O"},
]

def register_student():
    name = input("Enter your name: ")
    student_id = input("Enter your unique ID: ")
    if student_id in students:
        print("ID already exists. Please use a different ID.")
    else:
        students[student_id] = {"name": name, "scores": [], "quizzes_taken": 0}
        print("Registration successful!")

def login_student():
    student_id = input("Enter your unique ID: ")
    if student_id in students:
        return student_id
    else:
        print("ID not found. Please register first.")
        return None

def login_admin():
    username = input("Admin username: ")
    password = input("Admin password: ")
    if username in admins and admins[username] == password:
        return True
    else:
        print("Invalid admin credentials.")
        return False

def admin_interface():
    if not login_admin():
        return

    while True:
        print("\nAdmin Interface")
        print("1. Add Question")
        print("2. View All Questions")
        print("3. View Student Progress")
        print("4. Reset Student Data")
        print("5. Exit Admin")

        choice = input("Enter your choice: ")
        if choice == "1":
            add_question()
        elif choice == "2":
            view_questions()
        elif choice == "3":
            view_student_progress()
        elif choice == "4":
            reset_student_data()
        elif choice == "5":
            break
        else:
            print("Invalid choice.")

def add_question():
    question = input("Enter the question: ")
    options = []
    for i in range(3):  # Assume 3 options for simplicity
        option = input(f"Enter option {i + 1}: ")
        options.append(option)
    answer = input("Enter the correct answer from the provided options: ")
    if answer in options:
        questions.append({"question": question, "options": options, "answer": answer})
        print("Question added successfully!")
    else:
        print("Error: The correct answer must be one of the provided options.")

def view_questions():
    for i, q in enumerate(questions):
        print(f"{i + 1}. {q['question']} (Answer: {q['answer']})")

def view_student_progress():
    print("\n--- Student Progress ---")
    for student_id, data in students.items():
        avg_score = sum(data['scores']) / data['quizzes_taken'] if data['quizzes_taken'] > 0 else 0
        print(f"ID: {student_id}, Name: {data['name']}, Quizzes Taken: {data['quizzes_taken']}, Scores: {data['scores']}, Average Score: {avg_score:.2f}")

def reset_student_data():
    student_id = input("Enter the student ID to reset: ")
    if student_id in students:
        students[student_id] = {"name": students[student_id]["name"], "scores": [], "quizzes_taken": 0}
        print(f"Data for student {student_id} has been reset.")
    else:
        print("Student ID not found.")

def generate_quiz():
    return random.sample(questions, min(5, len(questions)))

def take_quiz(student_id, quiz, time_limit=30):
    score = 0
    start_time = time.time()
    for q in quiz:
        if time.time() - start_time > time_limit:
            print("Time's up! Quiz ended.")
            break

        print(q["question"])
        for i, option in enumerate(q["options"]):
            print(f"{i + 1}. {option}")
        try:
            answer_index = int(input("Your answer (number): ")) - 1
            if q["options"][answer_index] == q["answer"]:
                score += 1
        except (IndexError, ValueError):
            print("Invalid input. Moving to the next question.")

    students[student_id]["scores"].append(score)
    students[student_id]["quizzes_taken"] += 1
    print(f"Your score: {score}/{len(quiz)}")

def show_leaderboard():
    print("\n--- Leaderboard ---")
    ranked_students = sorted(students.items(), key=lambda x: (sum(x[1]['scores']) / x[1]['quizzes_taken']) if x[1]['quizzes_taken'] > 0 else 0, reverse=True)
    for student_id, data in ranked_students:
        avg_score = sum(data['scores']) / data['quizzes_taken'] if data['quizzes_taken'] > 0 else 0
        total_score = sum(data['scores'])
        print(f"ID: {student_id}, Name: {data['name']}, Total Score: {total_score}, Average Score: {avg_score:.2f}")

def main():
    while True:
        print("\nOptions:")
        print("1. Register Student")
        print("2. Login Student")
        print("3. Admin Interface")
        print("4. Show Leaderboard")
        print("5. Exit")

        choice = input("Enter your choice: ")

        if choice == "1":
            register_student()
        elif choice == "2":
            student_id = login_student()
            if student_id:
                quiz = generate_quiz()
                take_quiz(student_id, quiz)
        elif choice == "3":
            admin_interface()
        elif choice == "4":
            show_leaderboard()
        elif choice == "5":
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
from datetime import datetime
import re

def process_user_data(data):
    if not isinstance(data, list):
        raise ValueError("Input data must be a list")

    return [transform_user(user, index) for index, user in enumerate(data)]

def validate_user(user, index):
    if not isinstance(user, dict):
        raise ValueError(f"User at index {index} must be a dictionary")
    
    name = user.get('name')
    email = user.get('email')
    birthdate = user.get('birthdate')
    
    if not name or not isinstance(name, str):
        raise ValueError(f"User at index {index} must have a valid 'name'")
    if not email or not isinstance(email, str) or not is_valid_email(email):
        raise ValueError(f"User at index {index} must have a valid 'email'")
    if not birthdate or not is_valid_date(birthdate):
        raise ValueError(f"User at index {index} must have a valid 'birthdate'")

def transform_user(user, index):
    validate_user(user, index)
    
    name = user['name']
    email = user['email']
    birthdate = user['birthdate']

    return {
        "fullName": " ".join([word.capitalize() for word in name.strip().lower().split()]),
        "email": email.lower(),
        "age": calculate_age(datetime.strptime(birthdate, "%Y-%m-%d")),
        "birthdate": birthdate
    }

def is_valid_email(email):
    # Improved regex for email validation
    regex = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    return bool(re.match(regex, email))

def is_valid_date(date_str):
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
        return True
    except ValueError:
        return False

def calculate_age(birthdate):
    today = datetime.today()
    age = today.year - birthdate.year
    if (today.month, today.day) < (birthdate.month, birthdate.day):
        age -= 1
    return age

# Example usage
users = [
    { 
        "name": "john doe",
        "email": "email@gmail.c",
        "birthdate": "1990-05-15"
    },
    {
        "name": "jane smith",
        "email": "jane.smith@example.com",
        "birthdate": "1985-12-10"
    },
]

try:
    processed_users = process_user_data(users)
    print(processed_users)
except ValueError as e:
    print(e)
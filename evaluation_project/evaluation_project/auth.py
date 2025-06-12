import json
import os

FILE_PATH = 'users.json'

def load_users():
    """Loads users from a JSON file."""
    if not os.path.exists(FILE_PATH):
        return {}
    with open(FILE_PATH, 'r') as file:
        return json.load(file)

def save_user(username, password):
    """Saves a new user to the JSON file."""
    users = load_users()
    if username in users:
        return False
    users[username] = password
    with open(FILE_PATH, 'w') as file:
        json.dump(users, file)
    return True

def check_user(username, password):
    """Checks if the username and password match."""
    users = load_users()
    return users.get(username) == password
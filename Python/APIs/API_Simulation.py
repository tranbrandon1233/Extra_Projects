# Example usage in a Flask app:
from flask import Flask, request, jsonify

import time
from functools import wraps

class RateLimitExceeded(Exception):
    """Custom exception for exceeding the rate limit."""
    pass

class APISecurity:
    """
    This class implements a simple API security system with usage limits.
    """

    def __init__(self):
        # Replace "test_key" with your actual API keys and configure their limits.
        self.api_keys = {
            "test_key": {"limit": 10, "remaining": 10, "reset_time": time.time()}
        }

    def authenticate(self, api_key):
        """
        Authenticates the API key.

        Args:
            api_key: The API key to authenticate.

        Returns:
            True if the API key is valid, False otherwise.
        """
        return api_key in self.api_keys

    def rate_limit(self, api_key):
        """
        Applies rate limiting to the API key.

        Args:
            api_key: The API key to rate limit.

        Raises:
            RateLimitExceeded: If the rate limit is exceeded.
        """
        user_data = self.api_keys.get(api_key)
        if not user_data:
            return

        current_time = time.time()
        if current_time > user_data["reset_time"]:
            user_data["remaining"] = user_data["limit"]
            user_data["reset_time"] = current_time + 60  # Reset every 60 seconds

        if user_data["remaining"] > 0:
            user_data["remaining"] -= 1
        else:
            raise RateLimitExceeded("Rate limit exceeded. Please try again later.")

    def secure_endpoint(self, func):
        """
        Decorator to secure an API endpoint.

        Args:
            func: The API endpoint function.

        Returns:
            The decorated function.
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            api_key = kwargs.get("api_key")
            if not api_key:
                return {"error": "API key is required."}, 401

            if not self.authenticate(api_key):
                return {"error": "Invalid API key."}, 401

            try:
                self.rate_limit(api_key)
            except RateLimitExceeded as e:
                return {"error": str(e)}, 429

            return func(*args, **kwargs)
        return wrapper
    
app = Flask(__name__)

api_security = APISecurity()

@api_security.secure_endpoint
def my_api_endpoint(api_key, data):
    """
    A simple API endpoint that requires authentication and rate limiting.

    Args:
        api_key: The API key for authentication.
        data: The data to process.

    Returns:
        A dictionary containing the processed data.
    """
    # Process the data here...
    return {"message": "Data processed successfully.", "data": data}


@app.route('/api/data', methods=['POST'])
def data_endpoint():
    api_key = request.headers.get('X-API-Key')
    data = request.get_json()
    return jsonify(my_api_endpoint(api_key=api_key, data=data))

if __name__ == '__main__':
    app.run(debug=True)
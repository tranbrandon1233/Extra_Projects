// First install the dependencies by running the following command in the terminal:
// npm install dotenv express nodemailer bcrypt path

require('dotenv').config();
const express = require('express');
const nodemailer = require('nodemailer');
const bcrypt = require('bcrypt');
const path = require('path');
const app = express();

// Middleware setup - IMPORTANT: These need to be before any routes
app.use(express.json());
app.use(express.urlencoded({ extended: true }));
app.use(express.static('public'));

// Add request logging middleware for debugging
app.use((req, _, next) => {
  console.log(`${req.method} ${req.path}`, req.body);
  next();
});

// Create a .env file in your project root with these variables:
// GMAIL_USER=your-email@gmail.com
// GMAIL_APP_PASSWORD=your-16-character-app-password

// Email configuration
const transporter = nodemailer.createTransport({
  service: 'gmail',
  auth: {
    user: process.env.GMAIL_USER,
    pass: process.env.GMAIL_APP_PASSWORD
  }
});

// Test the email configuration
transporter.verify((error, _) => {
  if (error) {
    console.error('Email configuration error:', error);
  } else {
    console.log('Server is ready to send emails');
  }
});

// Modified registration endpoint with better error handling
app.post('/api/register', async (req, res) => {
  try {
    console.log(req.body);
    const { email, name, birthday, password } = req.body;

    // Validate input
    if (!email || !name || !birthday || !password) {
      return res.status(400).json({ error: 'All fields are required' });
    }

    // Check if user already exists
    if (users.has(email)) {
      return res.status(400).json({ error: 'Email already registered' });
    }

    // Hash password
    const hashedPassword = await bcrypt.hash(password, 10);

    // Generate verification code
    const verificationCode = generateVerificationCode();
    verificationCodes.set(email, {
      code: verificationCode,
      expiresAt: Date.now() + 3600000 // Code expires in 1 hour
    });

    // Store user data (unverified)
    users.set(email, {
      name,
      birthday,
      password: hashedPassword,
      verified: false
    });

    // Send verification email
    try {
      await transporter.sendMail({
        from: process.env.GMAIL_USER,
        to: email,
        subject: 'Verify Your Account',
        text: `Your verification code is: ${verificationCode}`,
        html: `
          <h1>Account Verification</h1>
          <p>Thank you for registering! Your verification code is:</p>
          <h2 style="color: #4CAF50; font-size: 24px;">${verificationCode}</h2>
          <p>This code will expire in 1 hour.</p>
        `
      });
      // Send success response
      res.json({ message: 'Registration successful. Please check your email for verification code.' });
    } catch (emailError) {
      // If email fails, clean up the created user
      users.delete(email);
      verificationCodes.delete(email);
      console.error('Email sending error:', emailError);
      res.status(500).json({ error: 'Failed to send verification email. Please try again later.' });
    }
    // Send error response
  } catch (error) {
    console.error('Registration error:', error);
    res.status(500).json({ error: 'Server error. Please try again later.' });
  }
});



app.use(express.json());
app.use(express.static('public')); // Serve static files

// In-memory storage (replace with actual database in production)
const users = new Map();
const verificationCodes = new Map();


// Generate random 6-digit code
function generateVerificationCode() {
  return Math.floor(100000 + Math.random() * 900000).toString();
}

// User registration endpoint
app.post('/api/register', async (req, res) => {
  try {
    const { email, name, birthday, password } = req.body;

    // Validate input
    if (!email || !name || !birthday || !password) {
      return res.status(400).json({ error: 'All fields are required' });
    }

    // Check if user already exists
    if (users.has(email)) {
      return res.status(400).json({ error: 'Email already registered' });
    }

    // Hash password
    const hashedPassword = await bcrypt.hash(password, 10);

    // Generate verification code
    const verificationCode = generateVerificationCode();
    verificationCodes.set(email, {
      code: verificationCode,
      expiresAt: Date.now() + 3600000 // Code expires in 1 hour
    });

    // Store user data (unverified)
    users.set(email, {
      name,
      birthday,
      password: hashedPassword,
      verified: false
    });

    // Send verification email
    const mailOptions = {
      from: 'your-email@gmail.com',
      to: email,
      subject: 'Verify Your Account',
      text: `Your verification code is: ${verificationCode}`
    };

    await transporter.sendMail(mailOptions);
    res.json({ message: 'Registration successful. Please check your email for verification code.' });

  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Server error' });
  }
});

// Verify account endpoint
app.post('/api/verify', (req, res) => {
  const { email, code } = req.body;
  // Get user and verification data
  const storedVerification = verificationCodes.get(email);
  const user = users.get(email);

  // Check if the user exists
  if (!storedVerification || !user) {
    return res.status(400).json({ error: 'Invalid email' });
  }

  // Check if the verification code is valid
  if (storedVerification.expiresAt < Date.now()) {
    verificationCodes.delete(email);
    return res.status(400).json({ error: 'Verification code expired' });
  }

  // Check if the verification code is correct
  if (storedVerification.code !== code) {
    return res.status(400).json({ error: 'Invalid verification code' });
  }

  // Mark user as verified
  user.verified = true;
  users.set(email, user);
  verificationCodes.delete(email);

  res.json({ message: 'Account verified successfully' });
});

// Login endpoint
app.post('/api/login', async (req, res) => {
  const { email, password } = req.body;
  const user = users.get(email);

  // Check if the user exists
  if (!user) {
    return res.status(400).json({ error: 'User not found' });
  }

  // Check if the account is verified
  if (!user.verified) {
    return res.status(400).json({ error: 'Account not verified' });
  }
  // Check if the password is valid
  const validPassword = await bcrypt.compare(password, user.password);
  if (!validPassword) {
    return res.status(400).json({ error: 'Invalid password' });
  }

  // In a real application, you would generate and return a JWT token here
  res.json({
    message: 'Login successful',
    user: {
      email,
      name: user.name,
      birthday: user.birthday
    }
  });
});


// Update the frontend code with fixed verification functionality
frontendCode = `<!DOCTYPE html>
<html>
<head>
    <title>User Registration</title>
    <style>
        .container {
            max-width: 400px;
            margin: 0 auto;
            padding: 20px;
        }
        .form-group {
            margin-bottom: 15px;
        }
        .form-group label {
            display: block;
            margin-bottom: 5px;
        }
        .form-group input {
            width: 100%;
            padding: 8px;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        .hidden {
            display: none;
        }
        button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 15px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            width: 100%;
        }
        button:hover {
            background-color: #45a049;
        }
        .error {
            color: red;
            margin-top: 5px;
            font-size: 14px;
        }
        .loading {
            opacity: 0.5;
            pointer-events: none;
        }
    </style>
</head>
<body>
    <div class="container">
        <!-- Registration Form -->
        <div id="registrationForm">
            <h2>Create Account</h2>
            <form onsubmit="register(event)">
                <div class="form-group">
                    <label for="email">Email:</label>
                    <input type="email" id="email" required>
                    <div id="emailError" class="error"></div>
                </div>
                <div class="form-group">
                    <label for="name">Name:</label>
                    <input type="text" id="name" required>
                    <div id="nameError" class="error"></div>
                </div>
                <div class="form-group">
                    <label for="birthday">Birthday:</label>
                    <input type="date" id="birthday" required>
                    <div id="birthdayError" class="error"></div>
                </div>
                <div class="form-group">
                    <label for="password">Password:</label>
                    <input type="password" id="password" required>
                    <div id="passwordError" class="error"></div>
                </div>
                <button type="submit" id="registerButton">Register</button>
            </form>
        </div>

        <!-- Verification Form -->
        <div id="verificationForm" class="hidden">
            <h2>Verify Your Account</h2>
            <form onsubmit="verify(event)">
                <div class="form-group">
                    <label for="verificationCode">Enter verification code:</label>
                    <input type="text" id="verificationCode" required>
                    <div id="verificationError" class="error"></div>
                </div>
                <button type="submit" id="verifyButton">Verify</button>
            </form>
        </div>

        <!-- Login Form -->
        <div id="loginForm" class="hidden">
            <h2>Login</h2>
            <form onsubmit="login(event)">
                <div class="form-group">
                    <label for="loginEmail">Email:</label>
                    <input type="email" id="loginEmail" required>
                    <div id="loginEmailError" class="error"></div>
                </div>
                <div class="form-group">
                    <label for="loginPassword">Password:</label>
                    <input type="password" id="loginPassword" required>
                    <div id="loginPasswordError" class="error"></div>
                </div>
                <button type="submit" id="loginButton">Login</button>
            </form>
        </div>

        <!-- User Profile -->
        <div id="userProfile" class="hidden">
            <h2>User Profile</h2>
            <div id="profileInfo"></div>
            <button onclick="logout()">Logout</button>
        </div>
    </div>

    <script>
        let currentEmail = '';

        async function register(event) {
            event.preventDefault();
            
            // Clear previous errors
            clearErrors();
            
            // Disable form while submitting
            const form = event.target;
            const button = document.getElementById('registerButton');
            form.classList.add('loading');
            button.textContent = 'Registering...';

            const email = document.getElementById('email').value.trim();
            const name = document.getElementById('name').value.trim();
            const birthday = document.getElementById('birthday').value;
            const password = document.getElementById('password').value;

            try {
                const response = await fetch('/api/register', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        email,
                        name,
                        birthday,
                        password
                    })
                });

                const data = await response.json();
                
                if (response.ok) {
                    currentEmail = email;
                    document.getElementById('registrationForm').classList.add('hidden');
                    document.getElementById('verificationForm').classList.remove('hidden');
                    alert(data.message);
                } else {
                    showError('emailError', data.error);
                }
            } catch (error) {
                console.error('Registration error:', error);
                alert('Registration failed. Please try again later.');
            } finally {
                form.classList.remove('loading');
                button.textContent = 'Register';
            }
        }

        async function verify(event) {
            event.preventDefault();
            
            // Clear previous errors
            clearErrors();
            
            // Disable form while submitting
            const form = event.target;
            const button = document.getElementById('verifyButton');
            form.classList.add('loading');
            button.textContent = 'Verifying...';

            const code = document.getElementById('verificationCode').value.trim();

            try {
                const response = await fetch('/api/verify', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        email: currentEmail,
                        code: code
                    })
                });

                const data = await response.json();
                
                if (response.ok) {
                    document.getElementById('verificationForm').classList.add('hidden');
                    document.getElementById('loginForm').classList.remove('hidden');
                    document.getElementById('loginEmail').value = currentEmail;
                    alert(data.message);
                } else {
                    showError('verificationError', data.error);
                }
            } catch (error) {
                console.error('Verification error:', error);
                alert('Verification failed. Please try again later.');
            } finally {
                form.classList.remove('loading');
                button.textContent = 'Verify';
            }
        }

        async function login(event) {
            event.preventDefault();
            
            // Clear previous errors
            clearErrors();
            
            // Disable form while submitting
            const form = event.target;
            const button = document.getElementById('loginButton');
            form.classList.add('loading');
            button.textContent = 'Logging in...';

            const email = document.getElementById('loginEmail').value.trim();
            const password = document.getElementById('loginPassword').value;

            try {
                const response = await fetch('/api/login', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        email,
                        password
                    })
                });

                const data = await response.json();
                
                if (response.ok) {
                    document.getElementById('loginForm').classList.add('hidden');
                    document.getElementById('userProfile').classList.remove('hidden');
                    document.getElementById('profileInfo').innerHTML = \`
                        <p>Name: \${data.user.name}</p>
                        <p>Email: \${data.user.email}</p>
                        <p>Birthday: \${data.user.birthday}</p>
                    \`;
                } else {
                    // Changed this line to show login errors under the password field
                    showError('loginPasswordError', data.error);
                }
            } catch (error) {
                console.error('Login error:', error);
                alert('Login failed. Please try again later.');
            } finally {
                form.classList.remove('loading');
                button.textContent = 'Login';
            }
        }

        function logout() {
            document.getElementById('userProfile').classList.add('hidden');
            document.getElementById('loginForm').classList.remove('hidden');
            document.getElementById('loginEmail').value = '';
            document.getElementById('loginPassword').value = '';
            clearErrors();
        }

        function showError(elementId, message) {
            document.getElementById(elementId).textContent = message;
        }

        function clearErrors() {
            const errorElements = document.getElementsByClassName('error');
            for (let element of errorElements) {
                element.textContent = '';
            }
        }
    </script>
</body>
</html>`;


// Write the HTML file
const fs = require('fs');
const publicDir = path.join(__dirname, 'public');

// Create the public directory if it doesn't exist
if (!fs.existsSync(publicDir)) {
    fs.mkdirSync(publicDir);
}

// Write the HTML file
fs.writeFileSync(path.join(publicDir, 'index.html'), frontendCode);

// Start the server
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
    console.log(`Visit http://localhost:${PORT} to access the application`);
});
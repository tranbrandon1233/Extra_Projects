// Replace 'YOUR_API_KEY' with your actual OpenWeatherMap API key
const apiKey = 'API_KEY';
// Replace 'CITY_NAME' with the city for which you want the weather data
const city = 'CITY';
const apiUrl = `https://api.openweathermap.org/data/2.5/weather?q=${city}&appid=${apiKey}&units=metric`; // Use 'imperial' for Fahrenheit

async function getWeatherData() {
  try {
    const response = await fetch(apiUrl);
    if (!response.ok) {
      throw new Error('Network response was not ok');
    }
    const data = await response.json();
    console.log(data); // Check the structure of the data in the console

    // Extracting the relevant data
    const temperature = data.main.temp;
    const description = data.weather[0].description;
    const humidity = data.main.humidity;
    const windSpeed = data.wind.speed;

    // Display the data in your app
    document.getElementById('temperature').textContent = `Temperature: ${temperature}°C`;
    document.getElementById('description').textContent = `Weather: ${description}`;
    document.getElementById('humidity').textContent = `Humidity: ${humidity}%`;
    document.getElementById('windSpeed').textContent = `Wind Speed: ${windSpeed} m/s`;

  } catch (error) {
    console.error('Error fetching weather data:', error);
  }
}

// Call the function to get weather data
getWeatherData();
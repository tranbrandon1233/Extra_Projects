const fs = require('fs');  // File system module for file operations
const readline = require('readline/promises');  // Readline module for reading input from the console
const natural = require('natural');  // Natural language processing library
const { SentimentAnalyzer, PorterStemmer } = natural;  // Importing SentimentAnalyzer and PorterStemmer from natural

const analyzer = new SentimentAnalyzer("English", PorterStemmer, "afinn");  // Initializing the SentimentAnalyzer

// Function to get the sentiment of lyrics
async function getSentiment(lyrics) {
  const result = analyzer.getSentiment(lyrics.split(" "));  // Analyze the sentiment of the lyrics
  if (result > 0) return "positive";  // Positive sentiment
  if (result < 0) return "negative";  // Negative sentiment
  return "neutral";  // Neutral sentiment
}

// Function to find the lyrics file based on the song title
async function findLyricsFile(songTitle) {
  const files = await fs.promises.readdir('.');  // Read current directory
  const normalizedTitle = songTitle.toLowerCase().replace(/[^a-z0-9]/g, '');  // Normalize the song title
  for (const file of files) {
    const normalizedFile = file.toLowerCase().replace(/\.txt$/, '').replace(/[^a-z0-9]/g, '');  // Normalize the file name
    if (normalizedFile === normalizedTitle) {
      return file;  // Return the matched file
    }
  }
  return null;  // Return null if no match is found
}

// Function to fetch a random activity from the Bored API
async function fetchFromBoredAPI() {
  const response = await fetch("https://bored-api.appbrewery.com/random");  // Fetch from the Bored API
  const data = await response.json();  // Parse the response JSON
  return data.activity;  // Return the activity
}

// Function to fetch a random dog image
async function fetchDogImage() {
  const response = await fetch("https://dog.ceo/api/breeds/image/random");  // Fetch a random dog image
  const data = await response.json();  // Parse the response JSON
  return data.message;  // Return the image URL
}

// Function to fetch a random quote from the ZenQuotes API
async function fetchZenQuote() {
  const response = await fetch("https://zenquotes.io/api/random");  // Fetch a random quote
  const data = await response.json();  // Parse the response JSON
  return `${data[0].q} - ${data[0].a}`;  // Return the quote and author
}

// Main function to handle user input and display results
async function main() {
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  const songTitle = await rl.question("Enter the song title: ");  // Prompt user for song title
  rl.close();
  const lyricsFile = await findLyricsFile(songTitle);  // Find the lyrics file

  if (!lyricsFile) {
    console.log("Lyrics file not found.");  // Handle case where file is not found
    return;
  }

  const lyrics = await fs.promises.readFile(lyricsFile, 'utf-8');  // Read the lyrics file
  const sentiment = await getSentiment(lyrics);  // Get the sentiment of the lyrics
  console.log(`Sentiment of ${songTitle}: ${sentiment}`);  // Display the sentiment

  if (sentiment === "neutral") {
    const activity = await fetchFromBoredAPI();  // Fetch a random activity if sentiment is neutral
    console.log(`Recommended activity: ${activity}`);
  } else if (sentiment === "positive") {
    const dogImage = await fetchDogImage();  // Fetch a dog image if sentiment is positive
    console.log(`Here's a dog picture: ${dogImage}`);
  } else {
    const quote = await fetchZenQuote();  // Fetch a quote if sentiment is negative
    console.log(`Here's a quote: ${quote}`);
  }
}

// Run the main function
main();

const axios = require('axios');

async function getData() {
    try {
        const response = await axios.get('https://datasets-server.huggingface.co/rows?dataset=jamescalam%2Fyoutube-transcriptions&config=default&split=train&offset=0&length=100');
        const data = response.data;

        // Create an object to store the word frequency
        const wordFrequency = {};

        // Loop through the text column and extract words
        data.rows.forEach((row) => {
            const text = row.row.text;
            const titles = row.row.title;
            const words = text.split(/[,\. ]+/);

            // Remove case sensitivity
            words.forEach((word) => {
                word = word.toLowerCase();

                // Ignore words with less than 4 or more than 15 characters
                if (word.length < 4 || word.length > 15) {
                    return;
                }

                // Increment the frequency of the word
                if (!wordFrequency[word]) {
                    wordFrequency[word] = 1;
                } else {
                    wordFrequency[word]++;
                }
            });
        });

        // Sort the word frequency object by value (frequency)
        const sortedWordFrequency = Object.keys(wordFrequency).map((key) => {
            return {
                word: key,
                frequency: wordFrequency[key]
            };
        }).sort((a, b) => b.frequency - a.frequency);

        // Get the top 10 most frequent words
        const topWords = sortedWordFrequency.slice(0, 10);

        console.log(topWords);
    } catch (error) {
        console.error(error);
    }
}

getData();
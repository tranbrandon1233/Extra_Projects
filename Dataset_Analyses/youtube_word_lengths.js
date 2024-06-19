const axios = require('axios');

async function getData() {
    try {
        const response = await axios.get('https://datasets-server.huggingface.co/rows?dataset=jamescalam%2Fyoutube-transcriptions&config=default&split=train&offset=0&length=100');
        const data = response.data;

        // Create an object to store the word frequency
        let wordCount = 0;
        let totalWords = 0;

        // Loop through the text column and extract words
        data.rows.forEach((row) => {
            const text = row.row.text;
            const words = text.split(/[,\. ]+/);
            
            totalWords += words.length;

            // Remove case sensitivity
            words.forEach((word) => {
                wordCount += word.length;
            });
        });


        // Get the top 10 most frequent words

        console.log(wordCount/totalWords);
    } catch (error) {
        console.error(error);
    }
}

getData();
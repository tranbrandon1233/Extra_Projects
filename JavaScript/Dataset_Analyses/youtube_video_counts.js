const axios = require('axios');

async function getData() {
    try {
        const response = await axios.get('https://datasets-server.huggingface.co/rows?dataset=jamescalam%2Fyoutube-transcriptions&config=default&split=train&offset=0&length=100');
        const data = response.data;

        // Create an object to store the word frequency
        let videoCounts = {};
        let countedVideos = new Set();
        let videoLengths = {};

        data.rows.forEach(row => {
            let videoId = row.row.video_id;

            // store the length of the video
            if(!videoLengths[videoId] || row.row.end > videoLengths[videoId]) {
                videoLengths[videoId] = row.row.end;
            }
        });

        data.rows.forEach(row => {
            let videoId = row.row.video_id;
            let videoLength = videoLengths[videoId];

            if(videoLength >= 300 && !countedVideos.has(videoId)) { // check if video is at least 5 minutes long and not counted before
                countedVideos.add(videoId); // add video ID to the set of counted videos

                let date = new Date(row.row.published);
                let year = date.getFullYear();
                let month = date.getMonth() + 1; // getMonth() returns month index starting from 0

                // initialize year object if not present
                if(!videoCounts[year]) {
                    videoCounts[year] = {};
                }

                // initialize month count if not present
                if(!videoCounts[year][month]) {
                    videoCounts[year][month] = 0;
                }

                videoCounts[year][month]++;
            }
        });

        console.log(videoCounts);

    } catch (error) {
        console.error(error);
    }
}

getData();
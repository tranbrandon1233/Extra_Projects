async function getData() {
    try {
        const url = 'https://datasets-server.huggingface.co/rows?dataset=cais%2Fmmlu&config=abstract_algebra&split=test&offset=0&length=100';
        const response = await fetch(url);
        const data = await response.json();

        const answers = data.rows.map(row => row.row.answer).filter(answer => answer >= 0 && answer <= 3);
        const count = {};

        answers.forEach(answer => {
            if (answer in count) {
                count[answer]++;
            } else {
                count[answer] = 1;
            }
        });

        const sortedCount = Object.keys(count).map(key => ({ key, value: count[key] })).sort((a, b) => b.value - a.value);
        console.log(sortedCount);

    } catch(error) {
        console.error(error);
    }
}

getData();
async function getData() {
    try {
        const url = 'https://datasets-server.huggingface.co/rows?dataset=cais%2Fmmlu&config=abstract_algebra&split=test&offset=0&length=100';
        const response = await fetch(url);
        const data = await response.json();

        const answers = data.rows.map(row => row.row.answer); // Access the "row" object first and then the "answer" value
        const questions = data.rows.map(row => row.row.question); // Access the "row" object first and then the "answer" value
        const count = {};
        const questionCount = {};

        answers.forEach((answer, i) => {
            if (answer in count) {
                count[answer] += questions[i].length;
                questionCount[answer]++;
            } else {
                count[answer] = questions[i].length;
                questionCount[answer] = 1;
            }
        });

        // Convert the count object to an array of objects and sort it in descending order
        const sortedCount = Object.keys(count)
            .filter(key => key >= 0 && key <= 3) // Filter the answers to include only values between 0 and 3
            .map(key => ({ key, value: count[key] / questionCount[key] })) // Calculate the average length
            .sort((a, b) => b.value - a.value);

        console.log(sortedCount);

    } catch (error) {
        console.error(error);
    }
}

getData();
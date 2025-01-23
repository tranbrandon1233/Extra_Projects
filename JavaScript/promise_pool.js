async function promisePool(functions, n) {
    const start = Date.now();
    const times = [];
    let nextIndex = 0;

    async function enqueue() {
        if (nextIndex >= functions.length) return;
        const currentIndex = nextIndex++;
        await functions[currentIndex]();
        times[currentIndex] = Date.now() - start;
        await enqueue();
    }

    const promises = [];
    for (let i = 0; i < n && i < functions.length; i++) {
        promises.push(enqueue());
    }

    await Promise.all(promises);
    return [times, Math.max(...times)];
}

// Test cases
(async () => {
    let functions = [
        () => new Promise(res => setTimeout(res, 300)),
        () => new Promise(res => setTimeout(res, 400)),
        () => new Promise(res => setTimeout(res, 200))
    ];
    let n = 2;

    let result = await promisePool(functions, n);
    console.log(result); // Should output [[300, 400, 500], 500]

    functions = [
        () => new Promise(res => setTimeout(res, 300)),
        () => new Promise(res => setTimeout(res, 400)),
        () => new Promise(res => setTimeout(res, 200))
    ];
    n = 5;

    result = await promisePool(functions, n);
    console.log(result); // Should output [[300, 400, 200], 400]
})();
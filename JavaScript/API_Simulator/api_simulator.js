"use strict";
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
class EventBroker {
    constructor() {
        this.subscribers = {};
        this.retryAttempts = 3;
        this.backoffTime = 1000; // milliseconds
    }
    publish(eventType, message) {
        console.log(`Publishing event of type "${eventType}": ${message}`);
        if (this.subscribers[eventType]) {
            this.subscribers[eventType].forEach((callback) => {
                this.handleEventWithRetries(callback, message);
            });
        }
    }
    subscribe(eventType, callback) {
        if (!this.subscribers[eventType]) {
            this.subscribers[eventType] = [];
        }
        this.subscribers[eventType].push(callback);
    }
    handleEventWithRetries(callback, message) {
        return __awaiter(this, void 0, void 0, function* () {
            for (let attempt = 1; attempt <= this.retryAttempts; attempt++) {
                try {
                    yield callback(message);
                    console.log(`Event processed successfully on attempt ${attempt}`);
                    return;
                }
                catch (error) {
                    console.log(`Error processing event on attempt ${attempt}: ${error}`);
                    if (attempt < this.retryAttempts) {
                        yield this.backoff(attempt);
                    }
                    else {
                        console.log("Max retry attempts reached. Event processing failed.");
                    }
                }
            }
        });
    }
    backoff(attempt) {
        return __awaiter(this, void 0, void 0, function* () {
            const delay = this.backoffTime * attempt;
            console.log(`Backing off for ${delay} ms`);
            return new Promise((resolve) => setTimeout(resolve, delay));
        });
    }
}
class DataProcessingService {
    constructor(eventBroker) {
        this.eventBroker = eventBroker;
        this.eventBroker.subscribe('data', this.processData.bind(this));
    }
    processData(message) {
        return __awaiter(this, void 0, void 0, function* () {
            console.log(`DataProcessingService received message: ${message}`);
            yield this.simulateExternalApiCall();
        });
    }
    simulateExternalApiCall() {
        return __awaiter(this, void 0, void 0, function* () {
            console.log("Simulating external API call...");
            yield new Promise((resolve) => setTimeout(resolve, 2000)); // Simulate latency
            console.log("External API call completed");
        });
    }
}
// Simulation of the system operation
function simulate() {
    return __awaiter(this, void 0, void 0, function* () {
        const eventBroker = new EventBroker();
        const dataProcessingService = new DataProcessingService(eventBroker);
        eventBroker.publish('data', "Hello from Service 1");
        eventBroker.publish('data', "Hello from Service 2");
    });
}
simulate();

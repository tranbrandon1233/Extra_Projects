class EventBroker {
    private subscribers: { [key: string]: Function[] } = {};
    private retryAttempts: number = 3;
    private backoffTime: number = 1000; // milliseconds
  
    public publish(eventType: string, message: string) {
      console.log(`Publishing event of type "${eventType}": ${message}`);
      if (this.subscribers[eventType]) {
        this.subscribers[eventType].forEach((callback) => {
          this.handleEventWithRetries(callback, message);
        });
      }
    }
  
    public subscribe(eventType: string, callback: Function) {
      if (!this.subscribers[eventType]) {
        this.subscribers[eventType] = [];
      }
      this.subscribers[eventType].push(callback);
    }
  
    private async handleEventWithRetries(callback: Function, message: string) {
      for (let attempt = 1; attempt <= this.retryAttempts; attempt++) {
        try {
          await callback(message);
          console.log(`Event processed successfully on attempt ${attempt}`);
          return;
        } catch (error) {
          console.log(`Error processing event on attempt ${attempt}: ${error}`);
          if (attempt < this.retryAttempts) {
            await this.backoff(attempt);
          } else {
            console.log("Max retry attempts reached. Event processing failed.");
          }
        }
      }
    }
  
    private async backoff(attempt: number) {
      const delay = this.backoffTime * attempt;
      console.log(`Backing off for ${delay} ms`);
      return new Promise((resolve) => setTimeout(resolve, delay));
    }
  }
  
  class DataProcessingService {
    private eventBroker: EventBroker;
  
    constructor(eventBroker: EventBroker) {
      this.eventBroker = eventBroker;
      this.eventBroker.subscribe('data', this.processData.bind(this));
    }
  
    private async processData(message: string) {
      console.log(`DataProcessingService received message: ${message}`);
      await this.simulateExternalApiCall();
    }
  
    private async simulateExternalApiCall() {
      console.log("Simulating external API call...");
      await new Promise((resolve) => setTimeout(resolve, 2000)); // Simulate latency
      console.log("External API call completed");
    }
  }
  
  // Simulation of the system operation
  async function simulate() {
    const eventBroker = new EventBroker();
    const dataProcessingService = new DataProcessingService(eventBroker);
  
    eventBroker.publish('data', "Hello from Service 1");
    eventBroker.publish('data', "Hello from Service 2");
  }
  
  simulate();
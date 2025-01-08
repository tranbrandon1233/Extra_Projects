class TaskQueue {
    constructor() {
      this.queue = [];
      this.taskIdCounter = 0; // To ensure order for tasks with the same priority
    }
  
    // Adds a task to the queue with the specified priority
    addTask(task, priority = 0) {
      if (typeof task !== "function") {
        throw new Error("Task must be a callable entity (a function).");
      }
      this.queue.push({ id: this.taskIdCounter++, task, priority });
      this._heapifyUp();
      console.log(`Task added with priority ${priority}.`);
    }
  
    // Removes and returns the task with the highest priority
    removeTask() {
      if (this.isEmpty()) {
        throw new Error("No tasks to remove.");
      }
      const highestPriorityTask = this.queue[0];
      const lastTask = this.queue.pop();
      if (!this.isEmpty()) {
        this.queue[0] = lastTask;
        this._heapifyDown();
      }
      return highestPriorityTask.task;
    }
  
    // Processes the task with the highest priority
    processTask() {
      if (this.isEmpty()) {
        console.log("No tasks to process.");
        return;
      }
      const task = this.removeTask();
      task(); // Execute the task
      console.log("Task processed.");
    }
  
    // Heapify up to restore heap property after adding a task
    _heapifyUp() {
      let index = this.queue.length - 1;
      const current = this.queue[index];
      while (index > 0) {
        const parentIndex = Math.floor((index - 1) / 2);
        const parent = this.queue[parentIndex];
        
        // Compare priority, and break if the heap property is satisfied
        if (
          parent.priority > current.priority || 
          (parent.priority === current.priority && parent.id < current.id)
        ) {
          break;
        }
  
        // Swap with parent
        this.queue[index] = parent;
        this.queue[parentIndex] = current;
        index = parentIndex;
      }
    }
  
    // Heapify down to restore heap property after removing the highest-priority task
    _heapifyDown() {
      let index = 0;
      const length = this.queue.length;
      const current = this.queue[index];
      
      while (true) {
        const leftChildIndex = 2 * index + 1;
        const rightChildIndex = 2 * index + 2;
        let largestIndex = index;
  
        // Check left child
        if (
          leftChildIndex < length &&
          (this.queue[leftChildIndex].priority > this.queue[largestIndex].priority ||
            (this.queue[leftChildIndex].priority === this.queue[largestIndex].priority &&
              this.queue[leftChildIndex].id < this.queue[largestIndex].id))
        ) {
          largestIndex = leftChildIndex;
        }
  
        // Check right child
        if (
          rightChildIndex < length &&
          (this.queue[rightChildIndex].priority > this.queue[largestIndex].priority ||
            (this.queue[rightChildIndex].priority === this.queue[largestIndex].priority &&
              this.queue[rightChildIndex].id < this.queue[largestIndex].id))
        ) {
          largestIndex = rightChildIndex;
        }
  
        // If no swaps are needed, break
        if (largestIndex === index) break;
  
        // Swap with the largest child
        this.queue[index] = this.queue[largestIndex];
        this.queue[largestIndex] = current;
        index = largestIndex;
      }
    }
  
    // Check if the queue is empty
    isEmpty() {
      return this.queue.length === 0;
    }
  }
  
  // Example Usage
  const taskQueue = new TaskQueue();
  
  // Add tasks with different priorities
  taskQueue.addTask(() => console.log("Task 1 executed"), 2);
  taskQueue.addTask(() => console.log("Task 2 executed"), 1);
  taskQueue.addTask(() => console.log("Task 3 executed"), 3);
  taskQueue.addTask(() => console.log("Task 4 executed"), 2);
  taskQueue.addTask(() => console.log("Task 5 executed"), 1);
  
  // Process tasks
  taskQueue.processTask(); // Task 3 executed
  taskQueue.processTask(); // Task 1 executed
  taskQueue.processTask(); // Task 4 executed
  taskQueue.processTask(); // Task 2 executed
  taskQueue.removeTask(); // Remove Task 5
  taskQueue.processTask(); // No tasks to process.
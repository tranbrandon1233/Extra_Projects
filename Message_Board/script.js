// Initialize the threads array
let threads = [];

// Function to create a new thread
function createThread() {
  const threadName = document.getElementById("thread-name").value;
  const threadMessage = document.getElementById("thread-message").value;
  const threadImage = document.getElementById("thread-image").files[0];

  const newThread = {
    name: threadName,
    message: threadMessage,
    image: threadImage,
    comments: []
  };

  threads.unshift(newThread);
  renderThreads();
}

// Function to create a new comment
function createComment(threadIndex) {
  const commentName = document.getElementById(`comment-name-${threadIndex}`).value;
  const commentMessage = document.getElementById(`comment-message-${threadIndex}`).value;
  const commentImage = document.getElementById(`comment-image-${threadIndex}`).files[0];

  const newComment = {
    name: commentName,
    message: commentMessage,
    image: commentImage
  };

  threads[threadIndex].comments.push(newComment);
  renderThreads();
}

// Function to render the threads
function renderThreads() {
  const threadsContainer = document.getElementById("threads-container");
  threadsContainer.innerHTML = "";

  threads.forEach((thread, index) => {
    const threadDiv = document.createElement("div");
    threadDiv.className = "thread";

    const threadHeader = document.createElement("h2");
    threadHeader.textContent = thread.name;
    threadDiv.appendChild(threadHeader);

    const threadMessage = document.createElement("p");
    threadMessage.textContent = thread.message;
    threadDiv.appendChild(threadMessage);

    if (thread.image) {
      const threadImage = document.createElement("img");
      threadImage.src = URL.createObjectURL(thread.image);
      threadImage.onclick = function() {
        if (this.style.width === "100px") {
          this.style.width = "200px";
        } else {
          this.style.width = "100px";
        }
      };
      threadImage.style.width = "100px";
      threadImage.style.height = "auto";
      threadDiv.appendChild(threadImage);
    }

    const commentsContainer = document.createElement("div");
    commentsContainer.className = "comments-container";
    threadDiv.appendChild(commentsContainer);

    thread.comments.forEach((comment, commentIndex) => {
      const commentDiv = document.createElement("div");
      commentDiv.className = "comment";

      const commentName = document.createElement("h3");
      commentName.textContent = comment.name;
      commentDiv.appendChild(commentName);

      const commentMessage = document.createElement("p");
      commentMessage.textContent = comment.message;
      commentDiv.appendChild(commentMessage);

      if (comment.image) {
        const commentImage = document.createElement("img");
        commentImage.src = URL.createObjectURL(comment.image);
        commentImage.onclick = function() {
          if (this.style.width === "100px") {
            this.style.width = "200px";
          } else {
            this.style.width = "100px";
          }
        };
        commentImage.style.width = "100px";
        commentImage.style.height = "auto";
        commentDiv.appendChild(commentImage);
      }

      commentsContainer.appendChild(commentDiv);
    });

    const commentForm = document.createElement("form");
    commentForm.innerHTML = `
      <input type="text" id="comment-name-${index}" placeholder="Name">
      <input type="text" id="comment-message-${index}" placeholder="Message">
      <input type="file" id="comment-image-${index}">
      <button type="button" onclick="createComment(${index})">Comment</button>
    `;
    threadDiv.appendChild(commentForm);

    threadsContainer.appendChild(threadDiv);
  });
}

// Render the threads for the first time
renderThreads();
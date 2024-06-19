// Function to dynamically load content as the user scrolls down the page
function loadContentOnScroll(containerSelector, contentSelector, loadMoreSelector, callback) {
    // Get the container element
    const container = document.querySelector(containerSelector);
    
    // Get the load more element
    const loadMore = document.querySelector(loadMoreSelector);
    
    // Create an IntersectionObserver instance
    const observer = new IntersectionObserver((entries) => {
      // Check if the load more element is visible
      if (entries[0].isIntersecting) {
        // Load new content
        callback().then((newContent) => {
          // Append the new content to the container
          newContent.forEach((item) => {
            container.appendChild(item);
          });
          
          // Update the observer to observe the new load more element
          observer.unobserve(loadMore);
          observer.observe(loadMore);
        });
      }
    }, {
      // Set the threshold to 1 to load new content when the load more element is fully visible
      threshold: 1,
    });
    
    // Observe the load more element
    observer.observe(loadMore);
  }
  
  // Example usage:
  // Assume we have a container element with the class 'container', a load more element with the class 'load-more', and a function to load new content
  loadContentOnScroll('.container', '.content', '.load-more', async () => {
    // Simulate loading new content
    const newContent = await new Promise((resolve) => {
      setTimeout(() => {
        const items = [];
        for (let i = 0; i < 10; i++) {
          const item = document.createElement('div');
          item.textContent = `Item ${i + 1}`;
          item.classList.add('content');
          items.push(item);
        }
        resolve(items);
      }, 1000);
    });
    return newContent;
  });
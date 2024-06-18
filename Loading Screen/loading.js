document.addEventListener('DOMContentLoaded', function() {
    const icons = ['icon1.jpg', 'icon2.png']; // Add your icons' filenames here
    let currentIconIndex = 0;
    const loadingScreen = document.getElementById('loadingSplash');
    const loadingIcon = document.getElementById('loadingIcon');
  
    // Function to change the icon
    function changeIcon() {
      currentIconIndex = (currentIconIndex + 1) % icons.length;
      loadingIcon.src = icons[currentIconIndex];
      loadingIcon.onerror = () => {
        console.error(`Error loading image: ${icons[currentIconIndex]}`);
      };
    }
  
    // Change the icon every second
    const iconInterval = setInterval(changeIcon, 1000);
  
    // Check if all images and videos are loaded
    const resources = document.querySelectorAll('img, video');
    let loadedCount = 0;
  
    resources.forEach(resource => {
      if (resource.tagName.toLowerCase() === 'img') {
        resource.onload = () => {
          loadedCount++;
          if (loadedCount === resources.length) {
            clearInterval(iconInterval);
            loadingScreen.style.display = 'none';
          }
        };
  
        // If the image is already in cache and hence might not trigger onload
        if (resource.complete) {
          loadedCount++;
        }
      } else if (resource.tagName.toLowerCase() === 'video') {
        resource.onloadeddata = () => {
          loadedCount++;
          if (loadedCount === resources.length) {
            clearInterval(iconInterval);
            loadingScreen.style.display = 'none';
          }
        };
      }
    });
  
    // Double-check in case all images/videos are cached and loaded instantly
    if (loadedCount === resources.length) {
      clearInterval(iconInterval);
      loadingScreen.style.display = 'none';
    }
  });
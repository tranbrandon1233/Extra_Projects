// Get the input element
const input = document.getElementById('input');
// Add this to your existing JavaScript file or in a <style> tag in your HTML
const style = document.createElement('style');
style.textContent = `
  .color-text {
    display: inline-block;
    padding: 2px 5px;
    margin: 2px 0;
    cursor: pointer;
  }
  .color-text:hover {
    background-color: #f0f0f0;
  }
`;
document.head.appendChild(style);

// Add an event listener to the input element
input.addEventListener('change', handleFileSelect);

// Function to handle file selection
function handleFileSelect(event) {
  // Get the selected file
  const file = event.target.files[0];

  // Check if the file is an image
  if (!file.type.startsWith('image/')) {
    alert('Please select an image file');
    return;
  }

  // Create a new FileReader
  const reader = new FileReader();

  // Add an event listener to the FileReader
  reader.addEventListener('load', handleFileLoad);

  // Add an event listener for errors
  reader.addEventListener('error', () => {
    alert('Error loading the image');
  });

  // Read the file as a data URL
  reader.readAsDataURL(file);
}

// Function to handle file loading
function handleFileLoad(event) {
  // Get the loaded data URL
  const dataURL = event.target.result;

  // Create a new Image
  const image = new Image();

  // Add an event listener to the Image
  image.addEventListener('load', handleImageLoad);

  // Add an event listener for errors
  image.addEventListener('error', () => {
    alert('Error loading the image');
  });

  // Set the src attribute of the Image to the loaded data URL
  image.src = dataURL;
}

// Function to handle image loading
function handleImageLoad(event) {
  // Get the loaded Image
  const image = event.target;

  // Create a new Canvas
  const canvas = document.createElement('canvas');

  // Set the width and height attributes of the Canvas to the width and height of the Image
  canvas.width = image.width;
  canvas.height = image.height;

  // Get the 2D drawing context of the Canvas
  const context = canvas.getContext('2d');

    // Store the original image data
  window.originalImageData = context.getImageData(0, 0, canvas.width, canvas.height);
  // Draw the Image on the Canvas
  context.drawImage(image, 0, 0);

  // Get the pixel data of the Canvas
  const pixelData = context.getImageData(0, 0, canvas.width, canvas.height).data;

// Create an object to store the color counts
const colorCounts = {
  red: 0,
  orange: 0,
  yellow: 0,
  green: 0,
  blue: 0,
  purple: 0,
  pink: 0,
  brown: 0,
  black: 0,
  white: 0,
  gray: 0
};

// Iterate over the pixel data
for (let i = 0; i < pixelData.length; i += 4) {
  // Get the RGBA values of the current pixel
  const r = pixelData[i];
  const g = pixelData[i + 1];
  const b = pixelData[i + 2];
  const a = pixelData[i + 3];

  // If the pixel is transparent, skip it
  if (a === 0) {
    continue;
  }

  // Categorize the color and increment the corresponding count
  if (isRed(r, g, b)) {
    colorCounts.red++;
  } else if (isOrange(r, g, b)) {
    colorCounts.orange++;
  } else if (isYellow(r, g, b)) {
    colorCounts.yellow++;
  } else if (isGreen(r, g, b)) {
    colorCounts.green++;
  } else if (isBlue(r, g, b)) {
    colorCounts.blue++;
  } else if (isPurple(r, g, b)) {
    colorCounts.purple++;
  } else if (isPink(r, g, b)) {
    colorCounts.pink++;
  } else if (isBrown(r, g, b)) {
    colorCounts.brown++;
  } else if (isBlack(r, g, b)) {
    colorCounts.black++;
  } else if (isWhite(r, g, b)) {
    colorCounts.white++;
  } else {
    colorCounts.gray++;
  }
}

  // Calculate the total number of pixels
  const totalPixels = pixelData.length / 4;


  // Create a string to store the result
  let result = '';

  // Iterate over the color counts
  for (const color in colorCounts) {
    // Calculate the percentage of the image
    const percentage = (colorCounts[color] / totalPixels * 100).toFixed(2);
    if(percentage != 0){
      // Append the result to the string
      result += `<span class="color-text" data-color="${color}" onclick="toggleHighlighting('${color}')" onmouseover="highlightColor('${color}')" onmouseout="resetImage()">${color}: ${colorCounts[color]} (${percentage}%)</span><br>`;
    }
  }

  // Display the image
  const imageContainer = document.getElementById('image-container');
  imageContainer.innerHTML = '';
  imageContainer.appendChild(canvas);

  // Display the result
  const resultContainer = document.getElementById('result-container');
  resultContainer.innerHTML = result;

  // Create and display color blocks
  const colorBlocks = createColorBlocks(colorCounts);
  resultContainer.appendChild(colorBlocks);

  // Store the original image data
  const originalImageData = context.getImageData(0, 0, canvas.width, canvas.height);

  // Function to reset the image
  function resetImage() {
    // Put the original image data back into the Canvas
    context.putImageData(originalImageData, 0, 0);
  }
  
  // Add the resetImage function to the window object
  window.resetImage = resetImage;
}
// Define color classification functions
function isRed(r, g, b) {
  return r > 200 && g < 100 && b < 100;
}

function isOrange(r, g, b) {
  return r > 200 && g > 100 && g < 200 && b < 100;
}

function isYellow(r, g, b) {
  return r > 200 && g > 200 && b < 100;
}

function isGreen(r, g, b) {
  return r < 200 && g > 200 && b < 200;
}

function isBlue(r, g, b) {
  return r < 100 && g < 200 && b > 200;
}

function isPurple(r, g, b) {
  return r > 100 && r < 200 && g < 100 && b > 200;
}

function isPink(r, g, b) {
  return r > 200 && g < 200 && b > 150;
}

function isBrown(r, g, b) {
  return r > 100 && r < 200 && g < 150 && b < 100;
}

function isBlack(r, g, b) {
  return r < 50 && g < 50 && b < 50;
}

function isWhite(r, g, b) {
  return r > 230 && g > 230 && b > 230;
}

function isGray(r, g, b) {
  return Math.abs(r - g) < 20 && Math.abs(r - b) < 20 && Math.abs(g - b) < 20 && 
         r > 50 && r < 230 && 
         !(r < 100 && g < 200 && b > 200) && // not blue
         !(r > 100 && r < 200 && g < 100 && b > 200); // not purple
}

// Function to count pixels of each color
function countPixels() {
  const canvas = document.querySelector('canvas');
  const context = canvas.getContext('2d');
  const pixelData = context.getImageData(0, 0, canvas.width, canvas.height).data;

  const colorCounts = {
    red: 0,
    orange: 0,
    yellow: 0,
    green: 0,
    blue: 0,
    purple: 0,
    pink: 0,
    brown: 0,
    black: 0,
    white: 0,
    gray: 0,
  };

  for (let i = 0; i < pixelData.length; i += 4) {
    const r = pixelData[i];
    const g = pixelData[i + 1];
    const b = pixelData[i + 2];
    const a = pixelData[i + 3];

    if (a === 0) {
      continue;
    }

    let categorizedColor;
    if (isRed(r, g, b)) {
      categorizedColor = 'red';
    } else if (isOrange(r, g, b)) {
      categorizedColor = 'orange';
    } else if (isYellow(r, g, b)) {
      categorizedColor = 'yellow';
    } else if (isGreen(r, g, b)) {
      categorizedColor = 'green';
    } else if (isBlue(r, g, b)) {
      categorizedColor = 'blue';
    } else if (isPurple(r, g, b)) {
      categorizedColor = 'purple';
    } else if (isPink(r, g, b)) {
      categorizedColor = 'pink';
    } else if (isBrown(r, g, b)) {
      categorizedColor = 'brown';
    } else if (isBlack(r, g, b)) {
      categorizedColor = 'black';
    } else if (isWhite(r, g, b)) {
      categorizedColor = 'white';
    } else {
      categorizedColor = 'gray';
    }

    colorCounts[categorizedColor]++;
  }

  // Only display colors that exist in the image
  for (const color in colorCounts) {
    if (colorCounts[color] > 0) {
      const colorPercentage = (colorCounts[color] / (pixelData.length / 4)) * 100;
      console.log(`${color}: ${colorCounts[color]} pixels (${colorPercentage.toFixed(2)}%)`);
    }
  }
}

let highlightedColor = null; // Add a variable to keep track of the highlighted color
let clickedColor = null; // Add a variable to keep track of the clicked color

function toggleHighlighting(color) {
  if (clickedColor === color) {
    // If the same color is clicked again, reset the image
    resetImage();
    clickedColor = null;
  } else {
    // If a different color is clicked, reset the image and highlight the new color
    resetImage();
    clickedColor = color;
    highlightColor(color);
  }
}

function highlightColor(color) {
  const canvas = document.querySelector('canvas');
  const context = canvas.getContext('2d');
  const imageData = context.getImageData(0, 0, canvas.width, canvas.height);
  const data = imageData.data;

  for (let i = 0; i < data.length; i += 4) {
    const r = data[i];
    const g = data[i + 1];
    const b = data[i + 2];
    const a = data[i + 3];

    if (a === 0) {
      continue;
    }

    if (!matchesColor(r, g, b, color)) {
      data[i + 3] = 0; // Set alpha to 0 (transparent) for non-matching colors
    }
  }

  context.putImageData(imageData, 0, 0);
}


function createColorBlocks(colorCounts) {
  const blockContainer = document.createElement('div');
  blockContainer.id = 'color-block-container';
  blockContainer.style.display = 'flex';
  blockContainer.style.flexWrap = 'wrap';
  blockContainer.style.gap = '5px'; // Add spacing between blocks
  blockContainer.style.marginTop = '20px';

  for (const color in colorCounts) {
    if (colorCounts[color] > 0) {
      const block = document.createElement('div');
      block.style.width = `${Math.sqrt(colorCounts[color])}px`;
      block.style.height = `${Math.sqrt(colorCounts[color])}px`;
      block.style.backgroundColor = getAverageColor(color);
      block.title = `${color}: ${colorCounts[color]} pixels`;
      block.onclick = () => toggleHighlighting(color); // Add onclick event listener
      block.onmouseover = () => {
        if (!clickedColor) {
          highlightColor(color);
        }
      }; // Add onmouseover event listener
      block.onmouseout = () => {
        if (!clickedColor) {
          resetImage();
        }
      }; // Add onmouseout event listener
      blockContainer.appendChild(block);
    }
  }

  return blockContainer;
}

// Add the toggleHighlighting function to the window object
window.toggleHighlighting = toggleHighlighting;

function getAverageColor(color) {
  const canvas = document.querySelector('canvas');
  const context = canvas.getContext('2d');
  const imageData = context.getImageData(0, 0, canvas.width, canvas.height);
  const data = imageData.data;

  let totalR = 0, totalG = 0, totalB = 0, count = 0;

  for (let i = 0; i < data.length; i += 4) {
    const r = data[i];
    const g = data[i + 1];
    const b = data[i + 2];
    const a = data[i + 3];

    if (a === 0) continue;

    if (matchesColor(r, g, b, color)) {
      totalR += r;
      totalG += g;
      totalB += b;
      count++;
    }
  }

  if (count === 0) return '#000000';

  const avgR = Math.round(totalR / count);
  const avgG = Math.round(totalG / count);
  const avgB = Math.round(totalB / count);

  return `rgb(${avgR}, ${avgG}, ${avgB})`;
}

function matchesColor(r, g, b, color) {
  switch (color) {
    case 'white': return isWhite(r, g, b);
    case 'black': return isBlack(r, g, b);
    case 'red': return isRed(r, g, b);
    case 'orange': return isOrange(r, g, b);
    case 'yellow': return isYellow(r, g, b);
    case 'green': return isGreen(r, g, b);
    case 'blue': return isBlue(r, g, b);
    case 'purple': return isPurple(r, g, b);
    case 'pink': return isPink(r, g, b);
    case 'brown': return isBrown(r, g, b);
    case 'gray': return isGray(r, g, b);
    default: return false;
  }
}
// Add the highlightColor function to the window object
window.highlightColor = highlightColor;
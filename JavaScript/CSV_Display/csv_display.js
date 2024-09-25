// Function to create the table
function createTable(csvData) {
    // Split the CSV data into rows
    const rows = csvData.split('\n');

    // Create the table element
    const table = document.createElement('table');
    table.border = '1';

    // Create the table rows
    for (let i = 0; i < rows.length; i++) {
        const row = rows[i].split(',');
        const tableRow = document.createElement('tr');

        // Create the table cells
        for (let j = 0; j < row.length; j++) {
            const cell = document.createElement('td');
            cell.contentEditable = 'true'; // Make the cell editable
            cell.textContent = row[j];

            // If this is the first row, make the cells bold and red
            if (i === 0) {
                cell.style.background = 'red';
                cell.style.color = 'white';
                cell.style.fontWeight = 'bold';

                // Add a click event handler to sort the table when a header cell is clicked
                cell.onclick = function(event) {
                    const column = j;
                    const order = event.target.getAttribute('data-order') === 'asc' ? 'desc' : 'asc';
                    event.target.setAttribute('data-order', order);
                    sortTable(table, column, order);
                };
            }

            // Add the cell to the table row
            tableRow.appendChild(cell);
        }

        // Add the table row to the table
        table.appendChild(tableRow);
    }

    // Add the table to the page
    document.body.appendChild(table);
}

// Function to sort the table
function sortTable(table, column, order) {
    const rows = Array.from(table.rows);
    const headerRow = rows.shift();

    rows.sort(function(rowA, rowB) {
        const cellA = rowA.cells[column].textContent;
        const cellB = rowB.cells[column].textContent;

        if (order === 'asc') {
            return cellA.localeCompare(cellB);
        } else {
            return cellB.localeCompare(cellA);
        }
    });

    rows.unshift(headerRow);

    // Rebuild the table with the sorted rows
    table.innerHTML = '';
    rows.forEach(function(row) {
        table.appendChild(row);
    });
}

// Function to handle the file input change event
function handleFileInput(event) {
    // Get the selected file
    const file = event.target.files[0];

    // Check if a file was selected
    if (file) {
        // Create a FileReader object
        const reader = new FileReader();

        // Set the onload event handler
        reader.onload = function(event) {
            // Create the table with the CSV data
            createTable(event.target.result);
        };

        // Read the CSV file
        reader.readAsText(file);
    }
}

// Create the file input element
const fileInput = document.createElement('input');
fileInput.type = 'file';
fileInput.accept = '.csv';

// Set the onchange event handler
fileInput.onchange = handleFileInput;

// Add the file input to the page
document.body.appendChild(fileInput);
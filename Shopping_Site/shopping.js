// Importing necessary components and functions from other files
import getData from '../services/messageService'; // Function to fetch data from messaging service
import { useEffect, useState } from 'react'; // React hooks for state management and side effects
import Table from '@mui/material/Table'; // Material UI table components
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableContainer from '@mui/material/TableContainer';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';
import Paper from '@mui/material/Paper';
import './MessageGrid.css' // CSS styles for the component

// Exporting the MessageGrid component
export function MessageGrid(props) {
  // Initializing state variables using useState hook
  // data: to store the fetched data
  // isLoading: to track the loading state of the component
  const [data, setData] = useState([...props.data]);
  const [isLoading, setIsLoading] = useState(true);

  // Using useEffect hook to fetch data when the component mounts
  useEffect(() => {
    // Defining an async function to fetch data
    const fetchData = async () => {
      try {
        // Calling the getData function to fetch data from the messaging service
        const result = await getData();
        // Updating the data state with the fetched result
        setData(result);
        // Setting isLoading to false after data is fetched
        setIsLoading(false);
      } catch (error) {
        // Logging any error that occurs during data fetching
        console.error('Error fetching data:', error);
        // Setting isLoading to false even if an error occurs
        setIsLoading(false);
      }
    };

    // Calling the fetchData function
    fetchData();
  }, [props.data]); // Dependency array to re-run the effect when props.data changes

  // Logging the fetched data to the console
  console.log(data);

  // Returning the JSX for the component
  return (
    <div>
      {/* Displaying a loading message when isLoading is true */}
      {isLoading && <p>Loading...</p>}
      <div className="tableContainer">
        {/* Displaying the table only when data is available */}
        {data.length > 0 && (
          <TableContainer sx={{ maxWidth: 650 }} component={Paper}>
            <Table sx={{ maxWidth: 650 }} size="small" aria-label="a dense table">
              <TableHead>
                {/* Defining the table header with column names */}
                <TableRow sx={{
                  "& th": {
                    fontWeight: "bold",
                    color: "white",
                    backgroundColor: "gray"
                  }
                }}
                >
                  <TableCell>Name</TableCell>
                  <TableCell align="right">Day</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {/* Mapping over the data array to render table rows */}
                {data.map((d) => (
                  <TableRow
                    key={d.id} // Unique key for each row
                    sx={{ '&:last-child td, &:last-child th': { border: 0 } }}
                  >
                    <TableCell component="th" scope="row">
                      {/* Displaying the name data in the first column */}
                      {d.data.name}
                    </TableCell>
                    <TableCell align="right">
                      {/* Displaying the day data in the second column */}
                      {d.data.day}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        )}
        {/* Displaying a message when no data is available and isLoading is false */}
        {!isLoading && data.length === 0 && <p>No data available</p>}
      </div>
    </div>
  );
}
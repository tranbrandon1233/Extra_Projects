import React, { useState } from 'react';
import Papa from 'papaparse';
import { useTable, useFilters, useSortBy } from 'react-table';

function Table({ columns, data, setData, setFilterInputValue, filterInputValue, addRow }) {
  const {
    getTableProps,
    getTableBodyProps,
    headerGroups,
    rows,
    prepareRow,
  } = useTable(
    {
      columns,
      data,
    },
    useFilters,
    useSortBy
  );
  const [newRow, setNewRow] = useState({});

  return (
    <table {...getTableProps()} style={{ border: 'solid 1px blue' }}>
      <thead>
        {headerGroups.map((headerGroup) => (
          <tr {...headerGroup.getHeaderGroupProps()}>
            {headerGroup.headers.map((column) => (
              <th
                {...column.getHeaderProps(column.getSortByToggleProps())}
                style={{
                  borderBottom: 'solid 3px red',
                  background: 'aliceblue',
                  color: 'black',
                  fontWeight: 'bold',
                }}
              >
                {column.render('Header')}
                <span>
                  {column.isSorted
                    ? column.isSortedDesc
                      ? ' '
                      : ' '
                    : ''}
                </span>
                <div>
                  {column.canFilter ? (
                    <input
                      type="text"
                      value={filterInputValue[column.id]}
                      onChange={(e) => {
                        const value = e.target.value;
                        setFilterInputValue((prev) => ({ ...prev, [column.id]: value }));
                        const originalData = addRow.originalData;
                        const filteredData = originalData.filter((row) =>
                          String(row[column.id]).toLowerCase().includes(value.toLowerCase())
                        );
                        setData(filteredData);
                      }}
                    />
                  ) : null}
                </div>
              </th>
            ))}
          </tr>
        ))}
      </thead>
      <tbody {...getTableBodyProps()}>
        {rows.map((row) => {
          prepareRow(row);
          return (
            <tr {...row.getRowProps()}>
              {row.cells.map((cell) => (
                <td
                  {...cell.getCellProps()}
                  style={{
                    padding: '10px',
                    border: 'solid 1px gray',
                    background: 'papayawhip',
                  }}
                >
                  {cell.render('Cell')}
                </td>
              ))}
            </tr>
          );
        })}
        <tr>
          {columns.map((column) => (
            <td>
              <input
                type="text"
                value={newRow[column.id]}
                onChange={(e) => {
                  const value = e.target.value;
                  setNewRow((prev) => ({ ...prev, [column.id]: value }));
                }}
              />
            </td>
          ))}
        </tr>
        <tr>
          <td colSpan={columns.length}>
            <button
              onClick={() => {
                addRow.addRow(newRow);
                setNewRow({});
              }}
            >
              Add Row
            </button>
          </td>
        </tr>
      </tbody>
    </table>
  );
}

function App() {
  const [data, setData] = useState([]);
  const [columns, setColumns] = useState([]);
  const [filterInputValue, setFilterInputValue] = useState({});
  const [originalData, setOriginalData] = useState([]);

  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    Papa.parse(file, {
      header: true,
      dynamicTyping: true,
      complete: (results) => {
        const data = results.data;
        const columns = Object.keys(data[0]).map((key) => ({
          Header: key,
          accessor: key,
          id: key,
          canFilter: true,
        }));
        setData(data);
        setOriginalData(data);
        setColumns(columns);
      },
    });
  };

  const addRow = (newRow) => {
    setData((prev) => [...prev, newRow]);
    setOriginalData((prev) => [...prev, newRow]);
  };

  return (
    <div>
      <input type="file" onChange={handleFileUpload} />
      {columns.length > 0 && (
        <Table
          columns={columns}
          data={data}
          setData={setData}
          setFilterInputValue={setFilterInputValue}
          filterInputValue={filterInputValue}
          addRow={{ addRow, originalData }}
        />
      )}
    </div>
  );
}

export default App;
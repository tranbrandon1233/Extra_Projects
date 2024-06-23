import React from 'react';

const Block = ({ color, children, id }) => {
  return (
    <div
      style={{
        backgroundColor: color,
        padding: '10px',
        margin: '10px',
        borderRadius: '10px',
        border: '1px solid black',
        cursor: 'move',
      }}
      id={id}
    >
      {children}
    </div>
  );
};

export default Block;
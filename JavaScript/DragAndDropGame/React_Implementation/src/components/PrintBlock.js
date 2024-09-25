import React from 'react';
import Block from './Block';

const PrintBlock = ({ id, message }) => {
  return (
    <Block color="#9C27B0" id={id}>
      <p>Print: {message}</p>
    </Block>
  );
};

export default PrintBlock;
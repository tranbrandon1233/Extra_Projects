import React from 'react';
import Block from './Block';

const LoopBlock = ({ id, iterations }) => {
  return (
    <Block color="#2196F3" id={id}>
      <p>Loop: {iterations} times</p>
    </Block>
  );
};

export default LoopBlock;
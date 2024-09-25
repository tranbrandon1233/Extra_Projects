import React from 'react';
import Block from './Block';

const ConditionalBlock = ({ id, condition }) => {
  return (
    <Block color="#FF9800" id={id}>
      <p>If: {condition}</p>
    </Block>
  );
};

export default ConditionalBlock;
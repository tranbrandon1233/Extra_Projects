import React from 'react';
import Block from './Block';

const VariableBlock = ({ id, variableName, variableValue }) => {
  return (
    <Block color="#4CAF50" id={id}>
      <p>Variable: {variableName} = {variableValue}</p>
    </Block>
  );
};

export default VariableBlock;
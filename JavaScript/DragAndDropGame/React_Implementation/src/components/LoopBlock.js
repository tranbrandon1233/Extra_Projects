import React from 'react';
import Block from './Block';
import { Droppable } from 'react-beautiful-dnd';

const LoopBlock = ({ id, iterations }) => {
  return (
    <Block color="#2196F3" id={id}>
      <p>Loop: {iterations} times</p>
      <Droppable droppableId={id}>
        {(provided) => (
          <div {...provided.droppableProps} ref={provided.innerRef}>
            {provided.placeholder}
          </div>
        )}
      </Droppable>
    </Block>
  );
};

export default LoopBlock;
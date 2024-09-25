import React, { useContext } from 'react';
import { Droppable, Draggable } from 'react-beautiful-dnd';
import CodeBlock from './Block';
import { DnDContext } from '../contexts/DnDContext';

const CodeBuilder = () => {
  const { codeBlocks } = useContext(DnDContext);

  return (
    <Droppable droppableId="code-builder">
      {(provided) => (
        <div {...provided.droppableProps} ref={provided.innerRef}>
          {codeBlocks.map((block, index) => (
            <Draggable key={block.id} draggableId={block.id} index={index}>
              {(provided) => (
                <div
                  {...provided.draggableProps}
                  {...provided.dragHandleProps}
                  ref={provided.innerRef}
                >
                  <CodeBlock color={block.color}>{block.content}</CodeBlock>
                </div>
              )}
            </Draggable>
          ))}
          {provided.placeholder}
        </div>
      )}
    </Droppable>
  );
};

export default CodeBuilder;
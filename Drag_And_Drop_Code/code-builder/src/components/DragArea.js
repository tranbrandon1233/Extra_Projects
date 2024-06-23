// components/DragArea.js

import React, { useState } from 'react';
import { DragDropContext, Droppable, Draggable } from 'react-beautiful-dnd';
import VariableBlock from './VariableBlock';
import ConditionalBlock from './ConditionalBlock';
import LoopBlock from './LoopBlock';
import PrintBlock from './PrintBlock';

const DragArea = () => {
  const [blocks, setBlocks] = useState([
    { id: 'variable-1', type: 'variable', variableName: 'x', value: 3 },
    { id: 'conditional-1', type: 'conditional', condition: 'x < 5' },
    { id: 'loop-1', type: 'loop', iterations: 5 },
    { id: 'print-1', type: 'print', message: 'Hello, World!' },
  ]);

  const onDragEnd = (result) => {
    if (!result.destination) return;
    const { source, destination } = result;
    const newBlocks = [...blocks];
    const [removed] = newBlocks.splice(source.index, 1);
    newBlocks.splice(destination.index, 0, removed);
    setBlocks(newBlocks);
  };

  const executeCode = () => {
    const x = blocks.find((block) => block.type === 'variable').value;
    if (x < 5) {
      console.log('Condition met!');
    } else {
      console.log('Condition not met!');
    }
    for (let i = 0; i < blocks.find((block) => block.type === 'loop').iterations; i++) {
      console.log(blocks.find((block) => block.type === 'print').message);
    }
  };

  return (
    <div>
      <DragDropContext onDragEnd={onDragEnd}>
        <Droppable droppableId="droppable-1">
          {(provided) => (
            <div
              {...provided.droppableProps}
              ref={provided.innerRef}
              style={{
                padding: '20px',
                width: '300px',
                border: '1px solid black',
              }}
            >
              {blocks.map((block, index) => (
                <Draggable key={block.id} draggableId={block.id} index={index}>
                  {(provided) => (
                    <div
                      ref={provided.innerRef}
                      {...provided.draggableProps}
                      {...provided.dragHandleProps}
                    >
                      {block.type === 'variable' && (
                        <VariableBlock
                          id={block.id}
                          variableName={block.variableName}
                          value={block.value}
                        />
                      )}
                      {block.type === 'conditional' && (
                        <ConditionalBlock id={block.id} condition={block.condition} />
                      )}
                      {block.type === 'loop' && (
                        <LoopBlock id={block.id} iterations={block.iterations} />
                      )}
                      {block.type === 'print' && (
                        <PrintBlock id={block.id} message={block.message} />
                      )}
                    </div>
                  )}
                </Draggable>
              ))}
              {provided.placeholder}
            </div>
          )}
        </Droppable>
      </DragDropContext>
      <button onClick={executeCode}>Execute Code</button>
    </div>
  );
};

export default DragArea;
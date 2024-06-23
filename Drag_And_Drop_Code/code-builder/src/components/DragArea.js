import React, { useState } from 'react';
import { DragDropContext, Droppable, Draggable } from 'react-beautiful-dnd';

const Block = ({ id, children, color }) => {
  return (
    <div style={{ backgroundColor: color, padding: '10px', border: '1px solid black' }}>
      {children}
    </div>
  );
};

const VariableBlock = ({ id, variableName, value, onChange }) => {
  return (
    <Block id={id} color="#ff69b4">
      <p>
        Variable {variableName} ={' '}
        <input type="number" value={value} onChange={(e) => onChange(e.target.value)} />
      </p>
    </Block>
  );
};

const ConditionalBlock = ({ id, condition, onChange }) => {
  return (
    <Block id={id} color="#33cc33">
      <p>
        If{' '}
        <input type="text" value={condition} onChange={(e) => onChange(e.target.value)} />
      </p>
    </Block>
  );
};

const LoopBlock = ({ id, iterations, onChange, children }) => {
  return (
    <Block id={id} color="#2196F3">
      <p>
        For Loop:{' '}
        <input type="number" value={iterations} onChange={(e) => onChange(e.target.value)} /> times
      </p>
      {children}
    </Block>
  );
};

const PrintBlock = ({ id, message, onChange }) => {
  return (
    <Block id={id} color="#ffff00">
      <p>
        Print:{' '}
        <input type="text" value={message} onChange={(e) => onChange(e.target.value)} />
      </p>
    </Block>
  );
};

const DragArea = () => {
  const [blocks, setBlocks] = useState([
    { id: 'variable-1', type: 'variable', variableName: 'x', value: 3 },
    { id: 'conditional-1', type: 'conditional', condition: 'x < 5' },
    {
      id: 'loop-1',
      type: 'loop',
      iterations: 5,
      children: [{ id: 'print-1', type: 'print', message: 'Hello, World!' }],
    },
  ]);

  const onDragEnd = (result) => {
    if (!result.destination) return;
    const { source, destination } = result;
    const newBlocks = [...blocks];
    const [removed] = newBlocks.splice(source.index, 1);
    newBlocks.splice(destination.index, 0, removed);
    setBlocks(newBlocks);
  };

  const onChange = (id, value) => {
    const newBlocks = [...blocks];
    const index = newBlocks.findIndex((block) => block.id === id);
    newBlocks[index].value = value;
    setBlocks(newBlocks);
  };

  const onConditionChange = (id, condition) => {
    const newBlocks = [...blocks];
    const index = newBlocks.findIndex((block) => block.id === id);
    newBlocks[index].condition = condition;
    setBlocks(newBlocks);
  };

  const onIterationsChange = (id, iterations) => {
    const newBlocks = [...blocks];
    const index = newBlocks.findIndex((block) => block.id === id);
    newBlocks[index].iterations = iterations;
    setBlocks(newBlocks);
  };

  const onMessageChange = (id, message) => {
    const newBlocks = [...blocks];
    const index = newBlocks.findIndex((block) => block.id === id);
    newBlocks[index].message = message;
    setBlocks(newBlocks);
  };

  const executeCode = () => {
    const variableBlock = blocks.find((block) => block.type === 'variable');
    const conditionalBlock = blocks.find((block) => block.type === 'conditional');
    const loopBlock = blocks.find((block) => block.type === 'loop');
    const printBlock = loopBlock.children[0];

    if (
      blocks.indexOf(variableBlock) < blocks.indexOf(conditionalBlock) &&
      blocks.indexOf(conditionalBlock) < blocks.indexOf(loopBlock)
    ) {
      const variableValue = variableBlock.value;
      const condition = conditionalBlock.condition;
      const iterations = loopBlock.iterations;
      const message = printBlock.message;

      if (eval(condition.replace('x', variableValue))) {
        for (let i = 0; i < iterations; i++) {
          console.log(message);
        }
      } else {
        console.log('Condition not met!');
      }
    } else {
      console.log('Blocks are not in the correct order!');
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
                      style={provided.draggableProps.style}
                    >
                      {block.type === 'variable' && (
                        <VariableBlock
                          id={block.id}
                          variableName={block.variableName}
                          value={block.value}
                          onChange={(value) => onChange(block.id, value)}
                        />
                      )}
                      {block.type === 'conditional' && (
                        <ConditionalBlock
                          id={block.id}
                          condition={block.condition}
                          onChange={(condition) => onConditionChange(block.id, condition)}
                        />
                      )}
                      {block.type === 'loop' && (
                        <LoopBlock
                          id={block.id}
                          iterations={block.iterations}
                          onChange={(iterations) => onIterationsChange(block.id, iterations)}
                        >
                          {block.children.map((child, index) => (
                            <PrintBlock
                              key={child.id}
                              id={child.id}
                              message={child.message}
                              onChange={(message) => onMessageChange(child.id, message)}
                            />
                          ))}
                        </LoopBlock>
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
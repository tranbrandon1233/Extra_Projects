import { createContext, useState } from 'react';
import { DragDropContext } from 'react-beautiful-dnd';

const DnDContext = createContext();

const DnDProvider = ({ children }) => {
  const [codeBlocks, setCodeBlocks] = useState([
    { id: '1', type: 'loop', color: 'blue', content: 'For Loop' },
    { id: '2', type: 'conditional', color: 'red', content: 'If Statement' },
    { id: '3', type: 'variable', color: 'green', content: 'Variable Declaration' },
  ]);

  const onDragEnd = (result) => {
    if (!result.destination) return;

    const newCodeBlocks = [...codeBlocks];
    const [removed] = newCodeBlocks.splice(result.source.index, 1);
    newCodeBlocks.splice(result.destination.index, 0, removed);

    setCodeBlocks(newCodeBlocks);
  };

  return (
    <DnDContext.Provider value={{ codeBlocks, onDragEnd }}>
      <DragDropContext onDragEnd={onDragEnd}>{children}</DragDropContext>
    </DnDContext.Provider>
  );
};

export { DnDProvider, DnDContext };
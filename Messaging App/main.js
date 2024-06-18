const socket = io();
const userId = Math.random().toString(36).substr(2, 9); // Generate a unique user ID for this session

const form = document.getElementById('chat-form');
const input = document.getElementById('message-input');
const messages = document.getElementById('messages');

form.addEventListener('submit', (e) => {
    e.preventDefault();
    if (input.value) {
        const message = {
            text: input.value,
            id: userId
        };
        socket.emit('chat message', message);
        addMessage(message, 'sent');
        input.value = '';
    }
});

socket.on('chat message', (message) => {
    if (message.id !== userId) {
        addMessage(message, 'received');
    }
});

function addMessage(message, type) {
    const item = document.createElement('div');
    item.classList.add('message', type);

    const bubble = document.createElement('div');
    bubble.classList.add('bubble');
    bubble.textContent = message.text;

    item.appendChild(bubble);
    messages.appendChild(item);
    messages.scrollTop = messages.scrollHeight;
}

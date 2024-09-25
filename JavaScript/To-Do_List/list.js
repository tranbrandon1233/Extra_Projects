document.addEventListener("DOMContentLoaded", function() {
    const createListBtn = document.getElementById('create-list-btn');
    const listTitleInput = document.getElementById('list-title-input');
    const addItemBtn = document.getElementById('add-item-btn');
    const itemInput = document.getElementById('item-input');
    const todoList = document.getElementById('todo-list');
    const titleContainer = document.getElementById('title-container');

    createListBtn.addEventListener('click', function() {
        const titleText = listTitleInput.value.trim();
        if (titleText !== "") {
            const title = document.createElement('h1');
            title.textContent = titleText;
            titleContainer.appendChild(title);
            listTitleInput.style.display = 'none';
            createListBtn.style.display = 'none';
        }
    });

    addItemBtn.addEventListener('click', function() {
        const itemText = itemInput.value.trim();
        if (itemText !== "") {
            const currentTime = new Date().toLocaleString();
            addItem(itemText, currentTime);
            itemInput.value = "";
        }
    });

    function addItem(text, createdTime) {
        const listItem = document.createElement('li');

        const checkbox = document.createElement('div');
        checkbox.className = 'checkbox';
        checkbox.addEventListener('click', function() {
            this.classList.toggle('checked');
            updateTooltip(listItem, createdTime);
        });

        const itemText = document.createElement('span');
        itemText.textContent = text;

        const infoIcon = document.createElement('span');
        infoIcon.className = 'info-icon';
        infoIcon.textContent = 'i';

        const tooltip = document.createElement('div');
        tooltip.className = 'tooltip';
        tooltip.textContent = `Created: ${createdTime}`;

        infoIcon.addEventListener('mouseover', function() {
            tooltip.style.display = 'block';
        });

        infoIcon.addEventListener('mouseout', function() {
            tooltip.style.display = 'none';
        });

        const removeBtn = document.createElement('button');
        removeBtn.className = 'remove-btn';
        removeBtn.textContent = 'Remove';
        removeBtn.addEventListener('click', function() {
            listItem.remove();
        });

        listItem.appendChild(checkbox);
        listItem.appendChild(itemText);
        const actionsContainer = document.createElement('div');
        actionsContainer.className = 'actions-container';
        const infoIconContainer = document.createElement('div');
        infoIconContainer.className = 'info-icon-container';
        infoIconContainer.appendChild(infoIcon);
        infoIconContainer.appendChild(tooltip);
        actionsContainer.appendChild(infoIconContainer);
        actionsContainer.appendChild(removeBtn);
        listItem.appendChild(actionsContainer);
        todoList.appendChild(listItem);
    }

    function updateTooltip(listItem, createdTime) {
        const tooltip = listItem.querySelector('.tooltip');
        const isChecked = listItem.querySelector('.checkbox').classList.contains('checked');
        const checkedTime = isChecked ? `Checked: ${new Date().toLocaleString()}` : '';
        tooltip.textContent = `Created: ${createdTime} ${checkedTime}`;
    }
});
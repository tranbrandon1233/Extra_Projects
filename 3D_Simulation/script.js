// script.js
let camera, scene, renderer, player, inventory, inventoryItems, raycaster, intersects, pickedObject, heldObject;

init();
animate();

function init() {
    camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 1, 1000);
    camera.position.y = 10;
    camera.position.z = 20; // move camera back to see objects

    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x87ceeb); // blue sky

    let floorGeometry = new THREE.PlaneGeometry(100, 100);
    let floorMaterial = new THREE.MeshBasicMaterial({ color: 0x32cd32 }); // green floor
    let floor = new THREE.Mesh(floorGeometry, floorMaterial);
    floor.rotation.x = -Math.PI / 2;
    scene.add(floor);

    player = new THREE.Mesh(new THREE.SphereGeometry(1, 32, 32), new THREE.MeshBasicMaterial({ color: 0xff0000 }));
    player.position.y = 1;
    scene.add(player);

    let object1 = new THREE.Mesh(new THREE.SphereGeometry(1, 32, 32), new THREE.MeshBasicMaterial({ color: 0x00ff00 }));
    object1.position.x = -10;
    scene.add(object1);

    let object2 = new THREE.Mesh(new THREE.SphereGeometry(1, 32, 32), new THREE.MeshBasicMaterial({ color: 0x0000ff }));
    object2.position.x = 10;
    scene.add(object2);

    let object3 = new THREE.Mesh(new THREE.SphereGeometry(1, 32, 32), new THREE.MeshBasicMaterial({ color: 0xffff00 }));
    object3.position.z = -10;
    scene.add(object3);

    let object4 = new THREE.Mesh(new THREE.SphereGeometry(1, 32, 32), new THREE.MeshBasicMaterial({ color: 0xff00ff }));
    object4.position.z = 10;
    scene.add(object4);

    let object5 = new THREE.Mesh(new THREE.SphereGeometry(1, 32, 32), new THREE.MeshBasicMaterial({ color: 0x00ffff }));
    object5.position.x = 5;
    object5.position.z = 5;
    scene.add(object5);

    let object6 = new THREE.Mesh(new THREE.SphereGeometry(1, 32, 32), new THREE.MeshBasicMaterial({ color: 0xffffff }));
    object6.position.x = -5;
    object6.position.z = 5;
    scene.add(object6);

    let object7 = new THREE.Mesh(new THREE.SphereGeometry(1, 32, 32), new THREE.MeshBasicMaterial({ color: 0x000000 }));
    object7.position.x = -5;
    object7.position.z = -5;
    scene.add(object7);

    let object8 = new THREE.Mesh(new THREE.SphereGeometry(1, 32, 32), new THREE.MeshBasicMaterial({ color: 0x999999 }));
    object8.position.x = 5;
    object8.position.z = -5;
    scene.add(object8);

    let object9 = new THREE.Mesh(new THREE.SphereGeometry(1, 32, 32), new THREE.MeshBasicMaterial({ color: 0x666666 }));
    object9.position.x = 0;
    object9.position.z = 10;
    scene.add(object9);

    inventory = document.getElementById('inventory');
    inventoryItems = [];

    for (let i = 0; i < 9; i++) {
        let item = document.createElement('div');
        item.className = 'inventory-item';
        item.innerHTML = '';
        inventory.appendChild(item);
        inventoryItems.push(item);
    }

    heldObject = document.getElementById('held-object');
    heldObject.style.display = 'none';

    raycaster = new THREE.Raycaster();
    intersects = [];

    renderer = new THREE.WebGLRenderer({
        antialias: true
    });
    renderer.setSize(window.innerWidth, window.innerHeight);
    document.body.appendChild(renderer.domElement);

    document.addEventListener('keydown', onKeyDown);
    document.addEventListener('keyup', onKeyUp);

    window.addEventListener('resize', onWindowResize);
}

function animate() {
    requestAnimationFrame(animate);

    raycaster.setFromCamera(new THREE.Vector2(0, 0), camera);
    intersects = raycaster.intersectObjects(scene.children);

    renderer.render(scene, camera);
}

function onKeyDown(event) {
    switch (event.key) {
        case 'w':
            player.position.z -= 1;
            camera.position.z -= 1; // move camera with player
            break;
        case 's':
            player.position.z += 1;
            camera.position.z += 1; // move camera with player
            break;
        case 'a':
            player.position.x -= 1;
            break;
        case 'd':
            player.position.x += 1;
            break;
        case 'e':
            pickUpObject();
            break;
        case '1':
        case '2':
        case '3':
        case '4':
        case '5':
        case '6':
        case '7':
        case '8':
        case '9':
            selectInventoryItem(event.key - 1);
            break;
    }
}

function onKeyUp(event) {
}

function onWindowResize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();

    renderer.setSize(window.innerWidth, window.innerHeight);
}

function pickUpObject() {
    if (intersects.length > 0) {
        let object = intersects[0].object;
        if (object !== player && object !== pickedObject) {
            let inventoryItem = getEmptyInventoryItem();
            if (inventoryItem) {
                inventoryItem.innerHTML = object.material.color.getHexString();
                heldObject.style.display = 'block';
                heldObject.innerHTML = object.material.color.getHexString();
                pickedObject = object;
                scene.remove(object);
            }
        }
    }
}

function getEmptyInventoryItem() {
    for (let i = 0; i < inventoryItems.length; i++) {
        if (inventoryItems[i].innerHTML === '') {
            return inventoryItems[i];
        }
    }
    return null;
}

function selectInventoryItem(index) {
    let item = inventoryItems[index];
    if (item) {
        for (let i = 0; i < inventoryItems.length; i++) {
            inventoryItems[i].classList.remove('active');
        }
        item.classList.add('active');
        heldObject.innerHTML = item.innerHTML;
    }
}
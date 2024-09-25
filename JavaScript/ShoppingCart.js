// Import the crypto-js library for secure encryption
const CryptoJS = require('crypto-js');

// Define the global window object
global.window = {
  localStorage: {
    data: {},
    getItem(key) {
      return this.data[key];
    },
    setItem(key, value) {
      this.data[key] = value;
    },
  },
};

class ShoppingCart {
  constructor(userId, passkey) {
    this.userId = userId;
    // Use crypto-js for secure encryption
    this.passkey = this.encryptPasskey(passkey);
    this.discountPercent = 10;
    this.cart = this.loadCartFromLocalStorage(passkey);
  }

  addProduct(product) {
    if (!product.id || !product.name || !product.price || !product.quantity || product.price < 0 || product.quantity < 0) {
      throw new Error('Invalid product details');
    }
    const existingProduct = this.cart.find(item => item.id === product.id);
    if (existingProduct) {
      existingProduct.quantity += product.quantity;
    } else {
      this.cart.push(product);
    }
    this.saveCartToLocalStorage();
  }

  removeProduct(productId) {
    this.cart = this.cart.filter(product => product.id !== productId);
    this.saveCartToLocalStorage();
  }

  updateProductQuantity(productId, quantity) {
    const product = this.cart.find(item => item.id === productId);
    if (product) {
      product.quantity = quantity;
      if (quantity <= 0) {
        this.removeProduct(productId);
      }
    }
    this.saveCartToLocalStorage();
  }

  calculateTotal() {
    return this.cart.reduce((total, product) => {
      return total + (product.price * product.quantity);
    }, 0);
  }

  applyDiscount() {
    if (this.discountPercent < 0 || this.discountPercent > 100) {
      throw new Error('Invalid discount percentage');
    }
    const total = this.calculateTotal();
    return total - (total * (this.discountPercent / 100));
  }

  setDiscount(discountPercent) {
    if (discountPercent < 0 || discountPercent > 100) {
      throw new Error('Invalid discount percentage');
    }
    this.discountPercent = discountPercent;
  }

  restoreDefaultDiscount() {
    this.discountPercent = 10;
  }

  saveCartToLocalStorage() {
    window.localStorage.setItem(`shoppingCart_${this.userId}`, JSON.stringify(this.cart));
  }

  loadCartFromLocalStorage(passkey) {
    const savedCart = window.localStorage.getItem(`shoppingCart_${this.userId}`);
    if (savedCart != null) {
      // Decrypt the passkey before comparing it
      const decryptedPasskey = CryptoJS.AES.decrypt(this.passkey, 'secretkey').toString(CryptoJS.enc.Utf8);
      if (passkey === decryptedPasskey) {
        return JSON.parse(savedCart);
      } else {
        throw new Error('Invalid passkey');
      }
    }
    return [];
  }

  // Use crypto-js for secure encryption
  encryptPasskey(passkey) {
    return CryptoJS.AES.encrypt(passkey, 'secretkey').toString();
  }

  getPasskey() {
    return this.passkey;
  }

  emptyCart() {
    this.cart = [];
    this.saveCartToLocalStorage();
  }

  getCartSummary() {
    const total = this.calculateTotal();
    return `
      Cart Summary:
      ${this.cart.map(product => `${product.name} - Quantity: ${product.quantity}, Price: $${product.price}`).join('\n')}
      Total Price: $${total}
      Total Price after ${this.discountPercent}% discount: $${this.applyDiscount()}
    `;
  }
}

const cart = new ShoppingCart('user123', 'password123');

const product1 = {
  id: 1,
  name: 'Product 1',
  price: 10.99,
  quantity: 2,
};

const product2 = {
  id: 2,
  name: 'Product 2',
  price: 5.99,
  quantity: 1,
};

cart.addProduct(product1);
cart.addProduct(product2);


cart.updateProductQuantity(1, 3);


cart.removeProduct(2);

console.log(cart.getCartSummary());
// Test saving and loading the cart with the passkey
cart.saveCartToLocalStorage();
const loadedCart = cart.loadCartFromLocalStorage('password123');
console.log(loadedCart);

// Test loading the cart with an invalid passkey
try {
  const invalidLoadedCart = cart.loadCartFromLocalStorage('invalidpasskey');
  console.log(invalidLoadedCart);
} catch (error) {
  console.log(error.message); // Output: Invalid passkey
}
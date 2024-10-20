#include <iostream>
#include <string>

struct Node {
    std::string key;
    int value;
    Node* next;
    Node(std::string key, int value) : key(key), value(value), next(nullptr) {}
};

class HashTable {
private:
    int capacity;
    int size;
    Node** table;
    int hash(const std::string& key) const {
        int prime = 31; // Choosing a prime for better distribution
        int hash_value = 0;
        for (int i = 0; i < key.length(); ++i) {
            hash_value += key[i] * (prime ^ i);
        }
        return hash_value % capacity; // Modulo to fit within table size
    }
public:
    HashTable(int capacity = 10) : capacity(capacity), size(0) {
        table = new Node * [capacity];
        for (int i = 0; i < capacity; ++i) {
            table[i] = nullptr;
        }
    }
    ~HashTable() {
        for (int i = 0; i < capacity; ++i) {
            Node* node = table[i];
            while (node) {
                Node* next = node->next;
                delete node;
                node = next;
            }
        }
        delete[] table;
    }
    void insert(const std::string& key, int value) {
        int index = hash(key);
        Node* node = table[index];
        while (node) {
            if (node->key == key) {
                node->value = value;
                return;
            }
            node = node->next;
        }
        node = new Node(key, value);
        node->next = table[index];
        table[index] = node;
        ++size;
        if (size / (double)capacity > 0.7) {
            resize();
        }
    }
    void resize() {
        int new_capacity = capacity * 2;
        Node** new_table = new Node * [new_capacity];
        for (int i = 0; i < new_capacity; ++i) {
            new_table[i] = nullptr;
        }
        for (int i = 0; i < capacity; ++i) {
            Node* node = table[i];
            while (node) {
                int new_index = hash(node->key) % new_capacity;
                Node* new_node = new Node(node->key, node->value);
                new_node->next = new_table[new_index];
                new_table[new_index] = new_node;
                node = node->next;
            }
        }
        delete[] table;
        table = new_table;
        capacity = new_capacity;
    }
    int search(const std::string& key) const {
        for (Node* node = table[hash(key)]; node; node = node->next) {
            if (node->key == key) {
                return node->value;
            }
        }
        return -1;
    }
    void remove(const std::string& key) {
        int index = hash(key);
        Node* node = table[index];
        Node* prev = nullptr;
        while (node) {
            if (node->key == key) {
                if (prev) {
                    prev->next = node->next;
                }
                else {
                    table[index] = node->next;
                }
                delete node;
                --size;
                return;
            }
            prev = node;
            node = node->next;
        }
    }
};

// Test function
void testHashTable() {
    HashTable ht(10);

    // Test insert
    ht.insert("one", 1);
    ht.insert("two", 2);
    ht.insert("three", 3);
    std::cout << "Inserted 'one', 'two', and 'three'." << std::endl;

    // Test search
    std::cout << "Search 'one': " << ht.search("one") << " (Expected: 1)" << std::endl;
    std::cout << "Search 'two': " << ht.search("two") << " (Expected: 2)" << std::endl;
    std::cout << "Search 'three': " << ht.search("three") << " (Expected: 3)" << std::endl;
    std::cout << "Search 'four': " << ht.search("four") << " (Expected: -1)" << std::endl;

    // Test remove
    ht.remove("two");
    std::cout << "Removed 'two'." << std::endl;
    std::cout << "Search 'two': " << ht.search("two") << " (Expected: -1)" << std::endl;

    // Test resize
    ht.insert("four", 4);
    ht.insert("five", 5);
    ht.insert("six", 6);
    ht.insert("seven", 7);
    ht.insert("eight", 8);
    std::cout << "Inserted more elements to trigger resize." << std::endl;
    std::cout << "Search 'five': " << ht.search("five") << " (Expected: 5)" << std::endl;
    std::cout << "Search 'eight': " << ht.search("eight") << " (Expected: 8)" << std::endl;
}

int main() {
    testHashTable();
    return 0;
}
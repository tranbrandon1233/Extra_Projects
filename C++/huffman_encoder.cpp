#include <iostream>
#include <fstream>
#include <queue>
#include <unordered_map>
#include <vector>
#include <bitset>
#include <string>

// Node structure for Huffman tree
struct Node {
    char ch;
    int freq;
    Node *left, *right;

    Node(char c, int f, Node* l = nullptr, Node* r = nullptr)
        : ch(c), freq(f), left(l), right(r) {}
};

// Custom comparator for priority queue
struct Compare {
    bool operator()(Node* l, Node* r) {
        return l->freq > r->freq;
    }
};

class HuffmanCoding {
private:
    std::unordered_map<char, std::string> huffmanCodes;
    Node* root;

    /**
     * @brief Build Huffman Tree from the given text.
     * 
     * @param text The input text to build the Huffman Tree from.
     * @return Node* The root node of the Huffman Tree.
     */
    Node* buildHuffmanTree(const std::string& text) {
        std::unordered_map<char, int> freqMap;
        for (char c : text) freqMap[c]++;

        std::priority_queue<Node*, std::vector<Node*>, Compare> pq;
        for (auto& pair : freqMap) {
            pq.push(new Node(pair.first, pair.second));
        }

        while (pq.size() > 1) {
            Node* left = pq.top(); pq.pop();
            Node* right = pq.top(); pq.pop();
            pq.push(new Node('\0', left->freq + right->freq, left, right));
        }

        return pq.top();
    }

    /**
     * @brief Generate Huffman codes for each character in the tree.
     * 
     * @param root The root node of the Huffman Tree.
     * @param code The current Huffman code (default is empty string).
     */
    void generateCodes(Node* root, std::string code = "") {
        if (!root) return;
        if (root->ch != '\0') {
            huffmanCodes[root->ch] = code;
        }
        generateCodes(root->left, code + "0");
        generateCodes(root->right, code + "1");
    }

    /**
     * @brief Encode the input text using the generated Huffman codes.
     * 
     * @param text The input text to encode.
     * @return std::string The encoded text.
     */
    std::string encode(const std::string& text) {
        std::string encodedText = "";
        for (char c : text) {
            encodedText += huffmanCodes[c];
        }
        return encodedText;
    }

    /**
     * @brief Decode the encoded text using the Huffman Tree.
     * 
     * @param encodedText The encoded text to decode.
     * @return std::string The decoded text.
     */
    std::string decode(const std::string& encodedText) {
        std::string decodedText = "";
        Node* current = root;
        for (char bit : encodedText) {
            if (bit == '0') current = current->left;
            else current = current->right;

            if (!current->left && !current->right) {
                decodedText += current->ch;
                current = root;
            }
        }
        return decodedText;
    }

public:
    HuffmanCoding() : root(nullptr) {}

    /**
     * @brief Compress the input text using Huffman encoding.
     * 
     * @param text The input text to compress.
     * @return std::string The compressed (encoded) text.
     */
    std::string compress(const std::string& text) {
        root = buildHuffmanTree(text);
        generateCodes(root);
        return encode(text);
    }

    /**
     * @brief Decompress the encoded text using Huffman decoding.
     * 
     * @param encodedText The encoded text to decompress.
     * @return std::string The decompressed (decoded) text.
     */
    std::string decompress(const std::string& encodedText) {
        return decode(encodedText);
    }

    /**
     * @brief Read the content of a file.
     * 
     * @param filename The name of the file to read.
     * @return std::string The content of the file.
     */
    std::string readFile(const std::string& filename) {
        std::ifstream file(filename);
        if (!file) {
            throw std::runtime_error("Unable to open file");
        }
        return std::string((std::istreambuf_iterator<char>(file)),
                            std::istreambuf_iterator<char>());
    }

    /**
     * @brief Write the compressed data to a file.
     * 
     * @param filename The name of the file to write to.
     * @param compressedData The compressed data to write.
     */
    void writeCompressedFile(const std::string& filename, const std::string& compressedData) {
        std::ofstream file(filename, std::ios::binary);
        if (!file) {
            throw std::runtime_error("Unable to create output file");
        }

        // Write the Huffman tree structure
        std::string treeStructure;
        serializeTree(root, treeStructure);
        file << treeStructure << '\n';

        // Write the compressed data
        std::bitset<8> bits;
        int bitCount = 0;
        for (char c : compressedData) {
            bits[7 - bitCount] = (c == '1');
            bitCount++;
            if (bitCount == 8) {
                file.put(bits.to_ulong());
                bits.reset();
                bitCount = 0;
            }
        }
        if (bitCount > 0) {
            file.put(bits.to_ulong());
        }
    }

    /**
     * @brief Serialize the Huffman tree to a string.
     * 
     * @param node The current node of the Huffman tree.
     * @param result The string to store the serialized tree.
     */
    void serializeTree(Node* node, std::string& result) {
        if (!node) return;
        if (!node->left && !node->right) {
            result += "1";
            result += node->ch;
        } else {
            result += "0";
            serializeTree(node->left, result);
            serializeTree(node->right, result);
        }
    }

    /**
     * @brief Read the compressed data from a file and decompress it.
     * 
     * @param filename The name of the file to read from.
     * @return std::string The decompressed (decoded) text.
     */
    std::string readCompressedFile(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file) {
            throw std::runtime_error("Unable to open compressed file");
        }

        // Read and deserialize the Huffman tree
        std::string treeStructure;
        getline(file, treeStructure);
        int index = 0;
        root = deserializeTree(treeStructure, index);

        // Read the compressed data
        std::string compressedData;
        char c;
        while (file.get(c)) {
            compressedData += std::bitset<8>(c).to_string();
        }

        // Decompress and return
        return decompress(compressedData);
    }

    /**
     * @brief Deserialize the Huffman tree from a string.
     * 
     * @param data The string containing the serialized tree.
     * @param index The current index in the string.
     * @return Node* The root node of the deserialized Huffman tree.
     */
    Node* deserializeTree(const std::string& data, int& index) {
        if (index >= data.length()) return nullptr;
        if (data[index] == '1') {
            index++;
            return new Node(data[index++], 0);
        }
        index++;
        // Get the node from the left and right subtrees
        Node* left = deserializeTree(data, index);
        Node* right = deserializeTree(data, index);
        // Return the node
        return new Node('\0', 0, left, right);
    }
};

int main() {
    HuffmanCoding huffman;
    std::string input, compressedData, decompressedData;

    // Read input from user
    std::string fileInput;
    std::cout << "Enter filename: ";
    std::getline(std::cin, fileInput);
    try {
        input = huffman.readFile(fileInput);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    std::string filename = fileInput.substr(0, fileInput.find_last_of('.'));

    // Compress the data
    compressedData = huffman.compress(input);
    std::cout << "Compressed data: " << compressedData << std::endl;
    std::cout << "Compression ratio: " << (float)compressedData.length() / (input.length() * 8) << std::endl;

    // Write compressed data to file
    huffman.writeCompressedFile(filename+"_compressed.bin", compressedData);
    std::cout << "Compressed data written to file '" + filename+"_compressed.bin'" << std::endl;

    // Read compressed data from file and decompress
    decompressedData = huffman.readCompressedFile(filename+"_compressed.bin");
    std::cout << "Decompressed data: " << decompressedData << std::endl;

    // Verify data integrity
    if (input == decompressedData) {
        std::cout << "Compression and decompression successful!" << std::endl;
    } else {
        std::cout << "Error: Decompressed data does not match original input." << std::endl;
    }

    return 0;
}
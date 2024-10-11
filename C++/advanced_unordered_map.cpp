#include <atomic>
#include <memory>
#include <iostream>

struct atomic_unordered_map {
    ~atomic_unordered_map() {}

    inline bool contains(int val) {
        auto idx = table.index(val, false);
        if (idx != nullptr && idx->load(std::memory_order_relaxed) != true)
            return true;
        return false;
    }

    inline void emplace(int val) {
        auto idx = table.index(val, false);
        if (idx == nullptr || idx->load(std::memory_order_relaxed) == true)
            table.index(val, true)->store(false);
    }

    inline void erase(int val) {
        auto idx = table.index(val, false);
        if (idx != nullptr) {
            bool expected = false;
            if (idx->compare_exchange_strong(expected, true, std::memory_order_acquire)) {
                delete idx;
                table.set_index(val, nullptr);
            }
        }
    }

    void reserve(unsigned int size) {
        table.reserve(size);
    }

private:
    struct jump_table {
        std::atomic_bool* index(int idx, bool op) noexcept {
            if (op && data[idx] == nullptr) {
                data[idx] = new std::atomic_bool(false);
            }
            return data[idx];
        }

        void set_index(int idx, std::atomic_bool* value) noexcept {
            data[idx] = value;
        }

        void reserve(unsigned int size_) {
            size.store(size_, std::memory_order_relaxed);
            data = std::make_unique<std::atomic_bool*[]>(size_);
            for (unsigned int i = 0; i < size_; ++i) {
                data[i] = nullptr;
            }
        }

    private:
        std::unique_ptr<std::atomic_bool*[]> data;
        std::atomic<unsigned int> size{0};
    };

    jump_table table;
};

int main() {
    atomic_unordered_map my_map;

    my_map.reserve(100); // Reserving space for 100 elements

    my_map.emplace(2);
    my_map.emplace(84);

    std::cout << "Contains 2? " << my_map.contains(2) << "\n"; // Should print 1 (true)
    std::cout << "Contains 84? " << my_map.contains(84) << "\n"; // Should print 1 (true)
    std::cout << "Contains 50? " << my_map.contains(50) << "\n"; // Should print 0 (false)

    my_map.erase(2);
    std::cout << "Contains 2 after erase? " << my_map.contains(2) << "\n"; // Should print 0 (false)

    return 0;
}


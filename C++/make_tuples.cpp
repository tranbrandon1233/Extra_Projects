#include <iostream>
#include <string>
#include <memory>
#include <tuple>
#include <utility>

struct A
{
    int i;
    double d;
    std::string s;
    A(int i, double d, const std::string& s) : i(i), d(d), s(s) {}
};

template <typename T, typename Tuple>
T* make_new_from_tuple(Tuple&& tuple) {
  return std::apply(
    [](auto&&... args) { return std::make_unique<T>(std::forward<decltype(args)>(args)...).release(); },
    std::forward<Tuple>(tuple)
  );
}

int main() {
    auto aTuple = std::make_tuple(1, 1.5, std::string("Hello"));
    A* aPtr = make_new_from_tuple<A>(aTuple);
    
    // Print information from aPtr
    std::cout << "A object:" << std::endl;
    std::cout << "i_: " << aPtr->i << std::endl;
    std::cout << "d_: " << aPtr->d << std::endl;
    std::cout << "s_: " << aPtr->s << std::endl;
    
    // Complex example with unique_ptr
    struct B {
        std::unique_ptr<int> ptr;
        B(std::unique_ptr<int>&& p) : ptr(std::move(p)) {}
    };
    auto bTuple = std::make_tuple(std::make_unique<int>(42));
    B* bPtr = make_new_from_tuple<B>(std::move(bTuple));
    
    // Print information from bPtr
    std::cout << "\nB object:" << std::endl;
    std::cout << "ptr value: " << *(bPtr->ptr) << std::endl;
    
    // Don't forget to delete the dynamically allocated objects
    delete aPtr;
    delete bPtr;
    
    return 0;
}
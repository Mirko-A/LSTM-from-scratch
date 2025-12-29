#include <systemc>
#include <iostream>

SC_MODULE (hello)
{
    SC_CTOR (hello)
    {
        std::cout << "Hello\n";
    }
};

int sc_main(int argc, char∗ argv[])
{
    hello h("hello");
    return (0);
}

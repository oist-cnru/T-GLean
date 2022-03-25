#include <glean.h>
#include <cnpy.h>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Path to config file required as first argument" << std::endl;
        exit(EXIT_FAILURE);
    }

    GPvrnn* rnn = new GPvrnn(argv[1], "testing"); // using GLean interface

    rnn->testInitialize(-1); // load last saved epoch
    rnn->test(); // prior and posterior generation
}

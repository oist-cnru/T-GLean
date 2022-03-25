#include <glean.h>
#include <cnpy.h>
#include <chrono>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Path to config file required as first argument" << std::endl;
        exit(EXIT_FAILURE);
    }

    GPvrnn* rnn = new GPvrnn(argv[1]); // using GLean interface

    // rnn->train(); // internal training routine

    /* Custom training routine */
    rnn->trainInitialize(-1); // resume training
    const int bg_epochs = std::max(rnn->save_epochs() / 10, 1);
    const int bg_start = rnn->current_epoch() / bg_epochs;
    const int bg_end = rnn->max_epochs() / bg_epochs;
    double rec_loss = 0.0;
    double reg_loss = 0.0;
    int etime, ctime;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int iterb = bg_start; iterb < bg_end; iterb++) {
        auto t1 = std::chrono::high_resolution_clock::now();
        rnn->train(bg_epochs, rec_loss, reg_loss, false); // Note: includes iotime
        auto t2 = std::chrono::high_resolution_clock::now();
        etime = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
        ctime = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t0).count();
        printf("Epoch %d  time %.0f / %.3f  loss_total %.7f  loss_rec %.7f  loss_reg %.7f\n", (iterb+1)*bg_epochs, ctime/1000.0, (etime/1000.0)/bg_epochs, rec_loss+reg_loss, rec_loss, reg_loss);
    }
}

#include <glean.h>
#include <cnpy.h>
#include <chrono>

/*
 * Test the GLean pipeline: train-test-plan
 *
 * Notes:
 * GLean supports a different feature set from base pvrnn. Do NOT try to use any extensions apart from planning
 * By policy GLean only supports native-order (row-major) arrays that form a 3D tensor of shape (count, length, dims). Do NOT try to load Fortran (column-major) data
 * While GLean saves are interchangable with base pvrnn saves, it is NOT recommended to mix saves
 * Results will differ between pvrnn and GLean modes -- this is expected
 */

int main(int argc, char* argv[]) {
    const string config_file = "planner/tests/test_gpipeline.toml";  // initialize the model using a pvrnn TOML config, just to show it works
    GPvrnn* rnn = new GPvrnn(config_file, "training", 1); // Fixed RNG seed

    /* Training */
    std::cout << "**Starting training**" << std::endl;
    rnn->trainInitialize(0); // restart training
    const int bg_epochs = rnn->save_epochs() / 10;
    const int max_iter = 10;
    const int b_count = rnn->max_epochs() / bg_epochs;
    if (b_count <= 0 || b_count > 100000) {
        std::cerr << "***Check training configuration! Got b_count = " << b_count << std::endl;
        exit(EXIT_FAILURE);
    }
    double init_loss = 0.0;
    double rec_loss = 0.0;
    double reg_loss = 0.0;
    int etime = 0;
    int ctime = 0;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int iterb = 0; iterb < b_count; iterb++) {
        auto t1 = std::chrono::high_resolution_clock::now();
        rnn->train(bg_epochs, rec_loss, reg_loss, false); // greedy train off
        auto t2 = std::chrono::high_resolution_clock::now();
        etime = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
        ctime = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t0).count();
        if (iterb == 0) init_loss = rec_loss + reg_loss;
        printf("Epoch %d  time %.1f / %.3f  loss_total %.7f  loss_rec %.7f  loss_reg %.7f\n",
               (iterb+1)*bg_epochs, ctime/1000.0, (etime/1000.0)/bg_epochs, rec_loss+reg_loss, rec_loss, reg_loss);
    }

    if (init_loss < rec_loss + reg_loss) {
        std::cerr << "***Training failed! Loss started at " << init_loss << " but ended at " << rec_loss + reg_loss << std::endl;
        exit(EXIT_FAILURE);
    }
    std::cout << "**Training finished**" << std::endl;

    /* Normally test() would be called here, but use the unit testing calls here instead */
    std::cout << "**Starting testing**" << std::endl;
    rnn->testInitialize(-1); // reinitialize for testing, load last saved epoch
    rnn->postGenAndSave();
    rnn->priGenAndSave(1);   // 1 step of posterior generation before prior generation (target regeneration)
    std::cout << "**Testing finished**" << std::endl;

    /* Plan generation */
    std::cout << "**Starting planning**" << std::endl;
    rnn = new GPvrnn(config_file, "planning", 1); // reload network with planning settings instead of ER settings
    // rnn->setData(_data); // TODO: remove the requirement to setData at test time
    rnn->planInitialize(-1); // reinitialize for planning, load last saved epoch
    const float goal_x = 0.35;
    const int max_planner_iterations = rnn->max_timesteps(); // Since we update the postdiction window every step, don't go beyond max timesteps
    if (max_planner_iterations <= 0 || max_planner_iterations > 1000) {
        std::cerr << "***Check planning configuration! Got max_planner_iterations = " << max_planner_iterations << std::endl;
        exit(EXIT_FAILURE);
    }
    /* Setup buffers */
    float *data_in = (float *)calloc(rnn->max_timesteps()*rnn->dims(), sizeof(float)); if (data_in == NULL) abort();
    float *plan = (float *)calloc(rnn->max_timesteps()*rnn->dims(), sizeof(float)); if (plan == NULL) abort();
    for (int t = 0; t < rnn->max_timesteps(); t++) data_in[(t*rnn->dims())+2] = goal_x; // fix goal signal (we start at 0,0 already)
    rnn->setPlanInput(data_in); // "sensory data" is input
    rnn->setMask(); // use mask from config

    /* Planner */
    t0 = std::chrono::high_resolution_clock::now();
    for (int t = 0; t < max_planner_iterations; t++) {
        auto t1 = std::chrono::high_resolution_clock::now();
        rnn->plan(rec_loss, reg_loss, to_string(t)); // run planner
        if (t == 0) init_loss = rec_loss + reg_loss;
        rnn->getPlanOutput(plan); // get plan
        data_in[(t*rnn->dims())] = plan[(t*rnn->dims())]; data_in[(t*rnn->dims())+1] = plan[(t*rnn->dims())+1]; // update input with plan values
        rnn->setPlanInput(data_in); // sensory data is input
        rnn->unmask(t);
        auto t2 = std::chrono::high_resolution_clock::now();
        etime = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
        ctime = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t0).count();
        printf("Epoch %d  time %.1f / %.3f  loss_total %.7f  loss_rec %.7f  loss_reg %.7f\n", 
               (t+1)*rnn->max_epochs(), ctime/1000.0, (etime/1000.0)/rnn->max_epochs(), rec_loss+reg_loss, rec_loss, reg_loss);
    }

    std::cout << "Final plan step: [" << plan[(rnn->max_timesteps()-1)*rnn->dims()] << ", " << plan[(rnn->max_timesteps()-1)*rnn->dims()+1] << ", " << plan[(rnn->max_timesteps()-1)*rnn->dims()+2] << "]" << endl;
    if (init_loss < rec_loss + reg_loss) {
        std::cerr << "***Planning failed! Loss started at " << init_loss << " but ended at " << rec_loss + reg_loss << std::endl;
        exit(EXIT_FAILURE);
    }
    if (plan[(rnn->max_timesteps()-1)*rnn->dims()] > goal_x+0.1 || plan[(rnn->max_timesteps()-1)*rnn->dims()] < goal_x-0.1) {
        std::cerr << "***Planning failed! Goal X was " << goal_x << " but ended at " << plan[(rnn->max_timesteps()-1)*rnn->dims()] << std::endl;
        exit(EXIT_FAILURE);
    }
    /* Collect network output */
    double *full_rec_err = (double *)malloc(sizeof(double)*rnn->window_length()); if (full_rec_err == NULL) abort();
    double *full_reg_err_l1 = (double *)malloc(sizeof(double)*rnn->window_length()); if (full_reg_err_l1 == NULL) abort();
    double *full_reg_err_l2 = (double *)malloc(sizeof(double)*rnn->window_length()); if (full_reg_err_l2 == NULL) abort();
    int n_zunits = 0;
    for (int l = 0; l < rnn->n_layers()-1; l++) n_zunits += rnn->z_units(l);
    float *Am = (float *)malloc(sizeof(float)*n_zunits*rnn->window_length()); if (Am == NULL) abort();
    float *As = (float *)malloc(sizeof(float)*n_zunits*rnn->window_length()); if (As == NULL) abort();
    rnn->getFullErRecErr(-1, full_rec_err);
    rnn->getFullErRegErr(0, -1, full_reg_err_l1);
    rnn->getFullErRegErr(1, -1, full_reg_err_l2);
    rnn->getPosteriorA(Am, As);
    std::cout << "Rec. error per step: ";
    for (int t = 0; t < rnn->window_length(); t++) {
        std::cout << full_rec_err[t] << " ";
        if (std::isnan(full_rec_err[t]) || std::abs(full_rec_err[t]) <= 1e-30f) {
            std::cerr << "***Unexpected value at " << t << std::endl;
            exit(EXIT_FAILURE);
        }
    }
    std::cout << std::endl;
    std::cout << "Reg. error per step: ";
    for (int t = 0; t < rnn->window_length(); t++) {
        std::cout << full_reg_err_l1[t]+full_reg_err_l2[t] << " ";
        if (std::isnan(full_reg_err_l1[t]+full_reg_err_l2[t]) || std::abs(full_reg_err_l1[t]) <= 1e-30f || std::abs(full_reg_err_l2[t]) <= 1e-30f) {
            std::cerr << "***Unexpected value at " << t << std::endl;
            exit(EXIT_FAILURE);
        }
    }
    std::cout << std::endl;
    std::cout << "A values: " << std::endl;
    n_zunits = 0;
    for (int l = rnn->n_layers()-2; l >= 0 ; l--) {
        for (int z = 0; z < rnn->z_units(l); z++) {
            printf("(%f %f) ", Am[n_zunits], As[n_zunits]);
            if (std::isnan(Am[n_zunits]+As[n_zunits]) || std::abs(Am[n_zunits]) <= 1e-30f || std::abs(As[n_zunits]) <= 1e-30f) {
                std::cerr << "***Unexpected value at " << n_zunits << std::endl;
                exit(EXIT_FAILURE);
            }
            n_zunits++;
        }
        std::cout << std::endl;
    }

    std::cout << "**Planning finished**" << std::endl;

    free(As); As = NULL;
    free(Am); Am = NULL;
    free(full_reg_err_l2); full_reg_err_l2 = NULL;
    free(full_reg_err_l1); full_reg_err_l1 = NULL;
    free(full_rec_err); full_rec_err = NULL;
    free(plan); plan = NULL;
    free(data_in); data_in = NULL;
}

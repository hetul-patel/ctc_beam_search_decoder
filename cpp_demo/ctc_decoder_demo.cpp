// C++ demo for ctc beam search decoder

# include <cmath>
# include <iostream>
# include "src/ctc_beam_search_decoder.h"

using namespace std;

int main()
{
    cout << "CTC Beam Search Decoder Cpp Demo" << endl;

    // Function Prototype
    // int ctc_beam_search_decoder(float* inputs, int* sequence_length, 
    //                         int beam_width, int top_paths, int max_time,
    //                         int batch_size, int num_classes,
    //                         int* decoded, float* log_probabilities, bool batch_first);


    // Prepare Inputs

    const int max_time      =  5;
    const int batch_size    =  2;
    const int num_classes   =  6;
    const int beam_width    = 50;
    const int top_paths     = 10;

    // float inputs[max_time][batch_size][num_classes] = {
    //         {{0.6, 0.0, 0.0, 0.4, 0.0, 0.0 }},
    //         {{0.0, 0.5, 0.0, 0.5, 0.0, 0.0 }},
    //         {{0.0, 0.4, 0.0, 0.6, 0.0, 0.0 }},
    //         {{0.0, 0.4, 0.0, 0.1, 0.0, 0.5 }},
    //         {{0.0, 0.5, 0.0, 0.5, 0.0, 0.0 }}
    //     };  

    // int sequence_length[batch_size] = {max_time};

    float inputs[max_time][batch_size][num_classes] = {
            {{0.6, 0.0, 0.0, 0.4, 0.0, 0.0 },{0.6, 0.0, 0.0, 0.4, 0.0, 0.0 }},
            {{0.0, 0.5, 0.0, 0.5, 0.0, 0.0 },{0.0, 0.5, 0.0, 0.5, 0.0, 0.0 }},
            {{0.0, 0.4, 0.0, 0.6, 0.0, 0.0 },{0.0, 0.4, 0.0, 0.6, 0.0, 0.0 }},
            {{0.0, 0.4, 0.0, 0.1, 0.0, 0.5 },{0.0, 0.4, 0.0, 0.1, 0.0, 0.5 }},
            {{0.0, 0.5, 0.0, 0.5, 0.0, 0.0 },{0.0, 0.5, 0.0, 0.5, 0.0, 0.0 }}
        };  

    int sequence_length[batch_size] = {max_time,max_time};
    
    // The CTCDecoder works with log-probs.
    int t, b, c;
    for (t = 0; t < max_time; ++t) {
        for (b = 0; b < batch_size; ++b) {
            for (c = 0; c < num_classes; ++c) {
                inputs[t][b][c] = std::log(inputs[t][b][c]);
            }
        }
    }

    // Prepare Outputs
    int decoded[top_paths][batch_size][max_time];
    float log_probabilities[batch_size][top_paths] = {{0.0f}};

    int status = -1;
    status = ctc_beam_search_decoder(   &inputs[0][0][0], 
                                        sequence_length,
                                        beam_width,
                                        top_paths,
                                        max_time,
                                        batch_size,
                                        num_classes,
                                        &decoded[0][0][0],
                                        &log_probabilities[0][0]);

    cout << "status : " << status << endl;
    if(status != 0)
        return 0;
        
    for (int path = 0; path < top_paths; ++path) {
        for (int b = 0; b < batch_size; ++b) {
            printf("\nPath_%d (prob = %.7f) = ",path,std::expf(log_probabilities[0][path]));
            for (int i = 0; i < max_time; ++i) {
                if(decoded[path][b][i] >= 0)
                    printf("%2d,",decoded[path][b][i]);
            }
        }
    }
    printf("\n");
    return 0;
}
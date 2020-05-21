
#include <vector>
#include <algorithm>
#include "ctc_beam_search_decoder.h"
#include "ctc_beam_scorer.h"
#include "ctc_beam_search.h"

extern "C" int ctc_beam_search_decoder(float* inputs, int* sequence_length, 
                            int beam_width, int top_paths, int max_time,
                            int batch_size, int num_classes,
                            int* decoded, float* log_probabilities)
{

    //Create Default beam scorer class object     
    ctc::CTCBeamSearchDecoder<float>::DefaultBeamScorer default_scorer;


    // Convert data containers to the format accepted by the decoder, simply
    // mapping the memory from the container to an Eigen::ArrayXi,::MatrixXf,
    // using Eigen::Map.
    Eigen::Map<const Eigen::ArrayXi> seq_len(&sequence_length[0], batch_size);
    std::vector<Eigen::Map<const Eigen::Matrix<float, 
                            Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>> eigen_inputs;
    eigen_inputs.reserve(max_time);
    for (int t = 0; t < max_time; ++t) {
        //eigen_inputs.emplace_back(&inputs[t][0][0], batch_size, num_classes);
        eigen_inputs.emplace_back(inputs, batch_size, num_classes);
        inputs += batch_size*num_classes;
    }


    // Define Output vector of shape: (top_paths x batchSize x None)
    // data type vector<vector<vector<int>>>
    // first vector of length  = top_paths (Decode method uses this to
    //                            decide top_k internally)
    // second vector of length = batch
    // third vector of dynamic length = string_length which is different
    // for each top_path
    std::vector<std::vector<std::vector<int>>> outputs(top_paths);
    for (typename std::vector<std::vector<int>>& output : outputs) {
        output.resize(batch_size);
    }

    
    // Define score matrix for top_paths of shape: (batch_size x top_paths)
    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> scores(
            log_probabilities, batch_size, top_paths);

    
    // Create CTC decoder object and pass default_scorer to it
    // second arg is beam_width which is 10 times the top_paths
    ctc::CTCBeamSearchDecoder<float> decoder(num_classes, beam_width, &default_scorer, batch_size);

    int status = -1;
    status = decoder.Decode(seq_len, eigen_inputs, &outputs, &scores);
    
    if(status != 0)
        return status;

    // Copy Paths to Output Matrix
    int path,batch;
    for (path = 0; path < top_paths; ++path) {
        for (batch = 0; batch < batch_size; ++batch){
            if(outputs[path][batch].size() < max_time)
                outputs[path][batch].resize(max_time,-1);
            std::copy(outputs[path][batch].begin(),outputs[path][batch].end(),decoded);
            decoded += max_time;
        }
    }

    return status;
}



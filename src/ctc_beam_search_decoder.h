#ifndef CUSTOM_CTC_CTC_BEAM_SEARCH_DECODER_H
#define CUSTOM_CTC_CTC_BEAM_SEARCH_DECODER_H

// Exposes ctc beam search decoded function to library users

// ------------------------------------------------------------------------------------
// Desc:
//      Performs beam search decoding on the logits given in input. It is a special 
//      case of the ctc_beam_search_decoder with top_paths=1 and beam_width=1 
//      (but that decoder is faster for this special case).
// Args:
//      inputs             : 3-D float array, size [max_time, batch_size, num_classes].
//                           The logits. For input of shape [batch_size, max_time, num_classes]
//                           set batch_first argument to true.
//      sequence_length    : 1-D int array containing sequence lengths, 
//                           aving size [batch_size].
//      beam_width         : An int scalar >= 0 (beam search beam width).
//      top_paths          : An int scalar >= 0, <= beam_width (controls output size).
//      max_time           : maximum time step in batch (0th dimension of input array).
//      batch_size         : total number of inputs (1st dimension of input array).
//      num_classes        : total num classes (blank char is at last index)
//      batch_first        : "true" if batch_size is left most dimension in input. "false" if
//                           batch_size is the second dimension in input.
// Returns:
//      decoded            : 3-D int array, size [top_paths, batch_size, max_time]
//      log_probabilities  : 2-D float array, size [batch_size, top_paths] containing
//                           containing sequence log-probabilities
// ------------------------------------------------------------------------------------
extern "C" int ctc_beam_search_decoder(float* inputs, int* sequence_length, 
                            int beam_width, int top_paths, int max_time,
                            int batch_size, int num_classes,
                            int* decoded, float* log_probabilities, bool batch_first = false);


#endif //CUSTOM_CTC_CTC_BEAM_SEARCH_DECODER_H

# Python demo for ctc beam search decoder

from ctc_beam_seach_decoder_pywraper import ctc_beam_search_decoder
import numpy as np
import time

print("-----------------------------------")
print("CTC Beam Search Decoder Python Demo")
print("-----------------------------------")

#--------------------------------------------------------------------------------------------
# Prepare inputs for ctc beam search decoder
#--------------------------------------------------------------------------------------------

# Inputs' shape must be : timesteps x batch_size x num_classes
# for e.g here [5,1,6]
inputs = np.array([
                    [[0.6, 0.0, 0.0, 0.4, 0.0, 0.0 ]],
                    [[0.0, 0.5, 0.0, 0.5, 0.0, 0.0 ]],
                    [[0.0, 0.4, 0.0, 0.6, 0.0, 0.0 ]],
                    [[0.0, 0.4, 0.0, 0.1, 0.0, 0.5 ]],
                    [[0.0, 0.5, 0.0, 0.5, 0.0, 0.0 ]],
                    ]).astype(np.float32)
# The CTCDecoder works with log-probs.
inputs = np.log(inputs)
# sequence_length shape must be [batch_size]
sequence_length = np.int32([inputs.shape[0]])


#--------------------------------------------------------------------------------------------
# Run decoder on input matrix
#--------------------------------------------------------------------------------------------
start   = time.time()
decoded, log_probabilities = ctc_beam_search_decoder(inputs, sequence_length,
                                                    beam_width=50, top_paths=10)
dur     = time.time() - start
print("Time %.4f ms"%(dur*1000))

#--------------------------------------------------------------------------------------------
# Run decoder on input matrix
#--------------------------------------------------------------------------------------------
for b in range(decoded.shape[1]): # batch_size
    print("Batch : %d"%(b+1))
    for n in range(decoded.shape[0]): # top_n paths
        path = decoded[n,b][np.where(decoded[n,b] != -1)[0]]
        print("Path %2d score = %.7f prob = %.3f Path ="%(n+1, log_probabilities[b,n], 
                                                        np.exp(log_probabilities[b,n])),
                                                        path)

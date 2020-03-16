# Python demo for ctc beam search decoder
import ctypes
import numpy as np
import numpy.ctypeslib as ctl

#--------------------------------------------------------------------------------------------
# Load ctc_beam_search_decoder shared library
#--------------------------------------------------------------------------------------------

# change lib_dir and lib_name to your own path
# lib extension varies base on os for e.g
# Linux     : libctc_beam_search_decoder.so
# Windows   : libctc_beam_search_decoder.dll
# maxOS     : libctc_beam_search_decoder.dylib
lib_dir  = "/Users/hetul/Documents/CTC/ctc_beam_search_decoder/ctc_beam_search_decoder/build/" 
lib_name = "libctc_beam_search_decoder.dylib"
ctc_lib = ctypes.CDLL(lib_dir+lib_name)


#--------------------------------------------------------------------------------------------
# get python handle for py_ctc_beam_search_decoder function
#--------------------------------------------------------------------------------------------

cfloatptr   = ctl.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS")
cintptr     = ctl.ndpointer(ctypes.c_int, flags="C_CONTIGUOUS")
cint        = ctypes.c_int

py_ctc_beam_search_decoder = ctc_lib.ctc_beam_search_decoder
py_ctc_beam_search_decoder.argtypes = [cfloatptr, cintptr, cint, cint, cint, cint, 
                                        cint, cintptr, cfloatptr]
          

# ------------------------------------------------------------------------------------
# Desc:
#      Performs beam search decoding on the logits given in input. It is a special 
#      case of the ctc_beam_search_decoder with top_paths=1 and beam_width=1 
#      (but that decoder is faster for this special case).
# Args:
#      inputs             : 3-D float array, size [max_time, batch_size, num_classes].
#                           The logits.
#      sequence_length    : 1-D int array containing sequence lengths, 
#                           aving size [batch_size].
#      beam_width         : An int scalar >= 0 (beam search beam width).
#      top_paths          : An int scalar >= 0, <= beam_width (controls output size).
# Returns:
#      decoded            : 3-D int array, size [top_paths, batch_size, max_time]
#      log_probabilities  : 2-D float array, size [batch_size, top_paths] containing
#                           containing sequence log-probabilities
# ------------------------------------------------------------------------------------
def ctc_beam_search_decoder(inputs, sequence_length, beam_width=100, top_paths=1):
    
    # prepare shape inputs
    max_time      = inputs.shape[0]
    batch_size    = inputs.shape[1]
    num_classes   = inputs.shape[2]
    
    # Allocate output matrix for storing decoded outputs
    decoded             = np.zeros((top_paths,batch_size,max_time),np.int32)
    log_probabilities   = np.zeros((batch_size,top_paths),np.float32)

    # Get Output from cpp function
    status = py_ctc_beam_search_decoder(inputs,
                                        sequence_length,
                                        beam_width,
                                        top_paths,
                                        max_time,
                                        batch_size,
                                        num_classes,
                                        decoded,
                                        log_probabilities)
    if status != 0:
        return None, None
    
    return decoded, log_probabilities

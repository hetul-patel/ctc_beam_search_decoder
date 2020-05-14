# CTC Beam Search Decoder

**CTC decoder | C++ implementation | Python implementation**

## Python

The following code-skeleton gives a first impression of how to use the decoding algorithm with Python. More details can be found in the python_demo dir.

```python
import numpy as np
from ctc_beam_seach_decoder_pywraper import ctc_beam_search_decoder

# Inputs' shape must be : timesteps x batch_size x num_classes
inputs = np.array([
                    [[0.6, 0.0, 0.0, 0.4, 0.0, 0.0 ]],
                    [[0.0, 0.5, 0.0, 0.5, 0.0, 0.0 ]],
                    [[0.0, 0.4, 0.0, 0.6, 0.0, 0.0 ]],
                    [[0.0, 0.4, 0.0, 0.1, 0.0, 0.5 ]],
                    [[0.0, 0.5, 0.0, 0.5, 0.0, 0.0 ]],
                    ]).astype(np.float32)
                    
# sequence_length shape must be [batch_size]
sequence_length = np.int32([inputs.shape[0]])

# decoded : [top_paths, batch_size, max_timestep]
# log_probabilities : [batch_size, top_paths]
decoded, log_probabilities = ctc_beam_search_decoder(inputs, sequence_length,
                                                 beam_width=50, top_paths=10)
```

## C++

The following code-skeleton gives a first impression of how to use the decoding algorithm with C++. More details can be found in the cpp_demo dir.

```c++
#include "src/ctc_beam_search_decoder.h"

// Prepare Inputs
const int max_time      =  5;
const int batch_size    =  1;
const int num_classes   =  6;
const int beam_width    = 50;
const int top_paths     = 10;

float inputs[max_time][batch_size][num_classes] = {
        {{0.6, 0.0, 0.0, 0.4, 0.0, 0.0 }},
        {{0.0, 0.5, 0.0, 0.5, 0.0, 0.0 }},
        {{0.0, 0.4, 0.0, 0.6, 0.0, 0.0 }},
        {{0.0, 0.4, 0.0, 0.1, 0.0, 0.5 }},
        {{0.0, 0.5, 0.0, 0.5, 0.0, 0.0 }}
    };  

int sequence_length[batch_size] = {max_time};

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
```

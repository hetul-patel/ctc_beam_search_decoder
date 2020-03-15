//
// Created by Hetul on 08/02/20.
//

#ifndef CUSTOM_CTC_CTC_BEAM_SCORER_H
#define CUSTOM_CTC_CTC_BEAM_SCORER_H

#include "ctc_beam_entry.h"

namespace ctc {
// Base implementation of a beam scorer used by default by the decoder that can
// be subclassed and provided as an argument to CTCBeamSearchDecoder, if complex
// scoring is required. Its main purpose is to provide a thin layer for
// integrating language model scoring easily.
template <typename T, typename CTCBeamState>
class BaseBeamScorer {
public:
    virtual ~BaseBeamScorer() {}
    // State initialization.
    virtual void InitializeState(CTCBeamState* root) const {}
    // ExpandState is called when expanding a beam to one of its children.
    // Called at most once per child beam. In the simplest case, no state
    // expansion is done.
    virtual void ExpandState(const CTCBeamState& from_state, int from_label,
                             CTCBeamState* to_state, int to_label) const {}
    // ExpandStateEnd is called after decoding has finished. Its purpose is to
    // allow a final scoring of the beam in its current state, before resorting
    // and retrieving the TopN requested candidates. Called at most once per beam.
    virtual void ExpandStateEnd(CTCBeamState* state) const {}
    // GetStateExpansionScore should be an inexpensive method to retrieve the
    // (cached) expansion score computed within ExpandState. The score is
    // multiplied (log-addition) with the input score at the current step from
    // the network.
    //
    // The score returned should be a log-probability. In the simplest case, as
    // there's no state expansion logic, the expansion score is zero.
    virtual T GetStateExpansionScore(const CTCBeamState& state,
                                     T previous_score) const {
        return previous_score;
    }
    // GetStateEndExpansionScore should be an inexpensive method to retrieve the
    // (cached) expansion score computed within ExpandStateEnd. The score is
    // multiplied (log-addition) with the final probability of the beam.
    //
    // The score returned should be a log-probability.
    virtual T GetStateEndExpansionScore(const CTCBeamState& state) const {
        return T(0);
    }
};

}  // namespace ctc

#endif //CUSTOM_CTC_CTC_BEAM_SCORER_H

//
// Created by Hetul on 09/02/20.
//

#ifndef CUSTOM_CTC_CTC_MATH_H
#define CUSTOM_CTC_CTC_MATH_H


template <typename T>
constexpr float kLogZero() {
    return -std::numeric_limits<T>::infinity();  // NOLINT
}

// Add logarithmic probabilities using:
// ln(a + b) = ln(a) + ln(1 + exp(ln(b) - ln(a)))
// The two inputs are assumed to be log probabilities.
// (GravesTh) Eq. 7.18
template <typename T>
inline T LogSumExp(T log_prob_1, T log_prob_2) {
    // const T kLogZero = -std::numeric_limits<T>::infinity();
    // Always have 'b' be the smaller number to avoid the exponential from
    // blowing up.
    if (log_prob_1 == kLogZero<T>()) {
        return log_prob_2;
    } else if (log_prob_2 == kLogZero<T>()) {
        return log_prob_1;
    } else {
        return (log_prob_1 > log_prob_2)
               ? log_prob_1 + log1pf(expf(log_prob_2 - log_prob_1))
               : log_prob_2 + log1pf(expf(log_prob_1 - log_prob_2));
    }
}

#endif //CUSTOM_CTC_CTC_MATH_H

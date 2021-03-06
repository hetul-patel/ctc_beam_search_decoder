//
// Created by Hetul on 08/02/20.
//

#ifndef CUSTOM_CTC_CTC_BEAM_ENTRY_H
#define CUSTOM_CTC_CTC_BEAM_ENTRY_H

#include <algorithm>
#include <limits>
#include <memory>
#include <vector>
#include <cmath>
#include <map>

#include "macros.h"
#include "ctc_math.h"

namespace ctc {

// The ctc_beam_search namespace holds several classes meant to be accessed only
// in case of extending the CTCBeamSearch decoder to allow custom scoring
// functions.
//
// BeamEntry is exposed through template arguments BeamScorer and BeamComparer
// of CTCBeamSearch (ctc_beam_search.h).
namespace ctc_beam_search {

struct EmptyBeamState {};

template <typename T>
struct BeamProbability {
    BeamProbability()
            : total(kLogZero<T>()), blank(kLogZero<T>()), label(kLogZero<T>()) {}
    void Reset() {
        total = kLogZero<T>();
        blank = kLogZero<T>();
        label = kLogZero<T>();
    }
    T total;
    T blank;
    T label;
};

template <class T, class CTCBeamState>
class BeamRoot;


template <class T, class CTCBeamState = EmptyBeamState>
struct BeamEntry {
    // BeamRoot<CTCBeamState>::AddEntry() serves as the factory method.
    friend BeamEntry<T, CTCBeamState>* BeamRoot<T, CTCBeamState>::AddEntry(
            BeamEntry<T, CTCBeamState>* p, int l);
    inline bool Active() const { return newp.total != kLogZero<T>(); }
    // Return the child at the given index, or construct a new one in-place if
    // none was found.
    BeamEntry<T, CTCBeamState>& GetChild(int ind) {
        auto entry = children.emplace(ind, nullptr);
        auto& child_entry = entry.first->second;
        // If this is a new child, populate the BeamEntry<CTCBeamState>*.
        if (entry.second) {
            child_entry = beam_root->AddEntry(this, ind);
        }
        return *child_entry;
    }
    
    std::vector<int> LabelSeq(bool merge_repeated) const {
        std::vector<int> labels;
        int prev_label = -1;
        const BeamEntry<T, CTCBeamState>* c = this;
        while (c->parent != nullptr) {  // Checking c->parent to skip root leaf.
            if (!merge_repeated || c->label != prev_label) {
                labels.push_back(c->label);
            }
            prev_label = c->label;
            c = c->parent;
        }
        std::reverse(labels.begin(), labels.end());
        return labels;
    }

    BeamEntry<T, CTCBeamState>* parent;
    int label;
    // All instances of child BeamEntry are owned by *beam_root.
    std::map<int, BeamEntry<T, CTCBeamState>*> children;
    BeamProbability<T> oldp;
    BeamProbability<T> newp;
    CTCBeamState state;

private:
    // Constructor giving parent, label, and the beam_root.
    // The object pointed to by p cannot be copied and should not be moved,
    // otherwise parent will become invalid.
    // This private constructor is only called through the factory method
    // BeamRoot<CTCBeamState>::AddEntry().
    BeamEntry(BeamEntry* p, int l, BeamRoot<T, CTCBeamState>* beam_root)
            : parent(p), label(l), beam_root(beam_root) {}
    BeamRoot<T, CTCBeamState>* beam_root;
    DISALLOW_COPY_AND_ASSIGN(BeamEntry);
};



// This class owns all instances of BeamEntry.  This is used to avoid recursive
// destructor call during destruction.
template <class T, class CTCBeamState = EmptyBeamState>
class BeamRoot {
public:
    BeamRoot(BeamEntry<T, CTCBeamState>* p, int l) {
        root_entry_ = AddEntry(p, l);
    }
    BeamRoot(const BeamRoot&) = delete;
    BeamRoot& operator=(const BeamRoot&) = delete;

    BeamEntry<T, CTCBeamState>* AddEntry(BeamEntry<T, CTCBeamState>* p, int l) {
        auto* new_entry = new BeamEntry<T, CTCBeamState>(p, l, this);
        beam_entries_.emplace_back(new_entry);
        return new_entry;
    }
    BeamEntry<T, CTCBeamState>* RootEntry() const { return root_entry_; }

private:
    BeamEntry<T, CTCBeamState>* root_entry_ = nullptr;
    std::vector<std::unique_ptr<BeamEntry<T, CTCBeamState>>> beam_entries_;
};

// BeamComparer is the default beam comparer provided in CTCBeamSearch.
template <class T, class CTCBeamState = EmptyBeamState>
class BeamComparer {
public:
    virtual ~BeamComparer() {}
    virtual bool inline operator()(const BeamEntry<T, CTCBeamState>* a,
                                   const BeamEntry<T, CTCBeamState>* b) const {
        return a->newp.total > b->newp.total;
    }
};

}  // namespace ctc_beam_search

}  // namespace ctc

#endif //CUSTOM_CTC_CTC_BEAM_ENTRY_H

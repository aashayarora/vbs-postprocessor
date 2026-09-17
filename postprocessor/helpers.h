// Self-contained C++ helpers for the candidate post-processor, loaded into PyROOT via
// ROOT.gInterpreter.Declare(open("helpers.h").read()). Copied verbatim from
// preselection/src/utils.cpp so the post-processor reproduces the exact candidate
// reconstruction of the (old) 1lep-cutbased selections, decoupled from the preselection
// build (only ROOT is needed — no correctionlib / ONNX / TMVA / CMSSW).
//
// NOTE: VBS jet tagging is NOT done here. The preselection already runs the VBS BDT for
// every channel and stores the tagged pair as all-jet-frame indices per variation
// (vbs_jet1_Jetidx[_<sfx>], vbs_jet2_Jetidx[_<sfx>], vbs_score[_<sfx>]). The post-processor
// rebuilds the VBS four-vectors from Jet_*[vbs_jet*_Jetidx_<sfx>] and only performs the
// boson-candidate assignment (boosted H/V, resolved di-jets, mlb).
#ifndef POSTPROCESS_HELPERS_H
#define POSTPROCESS_HELPERS_H

#include <string>
#include <cmath>

#include "ROOT/RVec.hxx"
#include "Math/Vector4D.h"
#include "TLorentzVector.h"

using ROOT::VecOps::RVec;

// min dR of each element of (vec_eta1, vec_phi1) to any of (vec_eta2, vec_phi2).
inline RVec<float> VVdR(const RVec<float>& vec_eta1, const RVec<float>& vec_phi1, const RVec<float>& vec_eta2, const RVec<float>& vec_phi2) {
    if (vec_eta1.empty()) return RVec<float>();
    if (vec_eta2.empty()) return RVec<float>(vec_eta1.size(), 999.0f);
    RVec<float> out(vec_eta1.size());
    for (size_t i = 0; i < vec_eta1.size(); i++) {
        float mindR = 999.;
        for (size_t j = 0; j < vec_eta2.size(); j++) {
            float dR = ROOT::VecOps::DeltaR(vec_eta1[i], vec_eta2[j], vec_phi1[i], vec_phi2[j]);
            if (dR < mindR) mindR = dR;
        }
        out[i] = mindR;
    }
    return out;
}

// dR of each element of (vec_eta, vec_phi) to a single object; 1.0 sentinel when obj is -999.
inline RVec<float> VdR(const RVec<float>& vec_eta, const RVec<float>& vec_phi, float obj_eta, float obj_phi) {
    RVec<float> out(vec_eta.size());
    if (obj_eta == -999 || obj_phi == -999) {
        std::fill(out.begin(), out.end(), 1.0f);
        return out;
    }
    for (size_t i = 0; i < vec_eta.size(); i++) {
        out[i] = ROOT::VecOps::DeltaR(vec_eta[i], obj_eta, vec_phi[i], obj_phi);
    }
    return out;
}

// invariant mass of each element of a collection with a single object.
inline RVec<float> VInvariantMass(const RVec<float>& vec_pt, const RVec<float>& vec_eta, const RVec<float>& vec_phi,
                                  const RVec<float>& vec_mass, float obj_pt, float obj_eta, float obj_phi, float obj_mass) {
    RVec<float> invMass(vec_pt.size());
    TLorentzVector obj1;
    obj1.SetPtEtaPhiM(obj_pt, obj_eta, obj_phi, obj_mass);
    for (size_t i = 0; i < vec_pt.size(); i++) {
        TLorentzVector obj2;
        obj2.SetPtEtaPhiM(vec_pt[i], vec_eta[i], vec_phi[i], vec_mass[i]);
        invMass[i] = (obj1 + obj2).M();
    }
    return invMass;
}

// all 2-combinations of indices of a collection; {{999},{999}} when fewer than 2.
inline RVec<RVec<int>> getJetPairs(const RVec<float>& goodJets) {
    if (goodJets.size() >= 2) {
        return ROOT::VecOps::Combinations(goodJets, 2);
    }
    RVec<RVec<int>> result;
    result.emplace_back(RVec<int>{999});
    result.emplace_back(RVec<int>{999});
    return result;
}

// pairwise invariant mass of two aligned jet lists (jet-pair kinematics).
inline RVec<float> VVInvariantMass(const RVec<float>& pt1, const RVec<float>& eta1, const RVec<float>& phi1, const RVec<float>& m1,
                                   const RVec<float>& pt2, const RVec<float>& eta2, const RVec<float>& phi2, const RVec<float>& m2) {
    RVec<float> invariant_mass;
    for (size_t i = 0; i < pt1.size(); ++i) {
        if (pt1[i] < 0 || pt2[i] < 0) { invariant_mass.push_back(-999.0f); continue; }
        auto vec1 = ROOT::Math::PtEtaPhiMVector(pt1[i], eta1[i], phi1[i], m1[i]);
        auto vec2 = ROOT::Math::PtEtaPhiMVector(pt2[i], eta2[i], phi2[i], m2[i]);
        invariant_mass.push_back((vec1 + vec2).M());
    }
    return invariant_mass;
}

// pairwise pT of two aligned jet lists.
inline RVec<float> VVInvariantPt(const RVec<float>& pt1, const RVec<float>& eta1, const RVec<float>& phi1, const RVec<float>& m1,
                                 const RVec<float>& pt2, const RVec<float>& eta2, const RVec<float>& phi2, const RVec<float>& m2) {
    RVec<float> ptjj;
    for (size_t i = 0; i < pt1.size(); ++i) {
        if (pt1[i] < 0 || pt2[i] < 0) { ptjj.push_back(-999.0f); continue; }
        auto vec1 = ROOT::Math::PtEtaPhiMVector(pt1[i], eta1[i], phi1[i], m1[i]);
        auto vec2 = ROOT::Math::PtEtaPhiMVector(pt2[i], eta2[i], phi2[i], m2[i]);
        ptjj.push_back((vec1 + vec2).Pt());
    }
    return ptjj;
}

// pairwise dR of two aligned jet lists.
inline RVec<float> VVDeltaR(const RVec<float>& eta1, const RVec<float>& phi1, const RVec<float>& eta2, const RVec<float>& phi2) {
    RVec<float> dR;
    for (size_t i = 0; i < eta1.size(); ++i) {
        if (eta1[i] < -900 || eta2[i] < -900) { dR.push_back(999.0f); continue; }
        float dphi = std::abs(phi1[i] - phi2[i]);
        if (dphi > M_PI) dphi = 2 * M_PI - dphi;
        float deta = eta1[i] - eta2[i];
        dR.push_back(std::sqrt(deta * deta + dphi * dphi));
    }
    return dR;
}

// Overlay per-good-jet scores (in good-jet order) back onto the all-jet score array at the
// good-jet positions. Used to propagate the preselection's QCD-resampled NOMINAL good-fat-jet
// HvsQCD/VvsQCD onto the all-fat-jet array, so any variation's good-jet slice keeps the
// resampled values (scores are JEC-invariant). good_scores must be all_scores masked by
// good_mask (i.e. FatJet_HvsQCD[FatJet_isGood]) before resampling.
template <typename M>
inline RVec<float> scatter_good_scores(RVec<float> all_scores, const RVec<float>& good_scores,
                                       const RVec<M>& good_mask) {
    std::size_t k = 0;
    for (std::size_t i = 0; i < all_scores.size() && k < good_scores.size(); ++i)
        if (good_mask[i]) all_scores[i] = good_scores[k++];
    return all_scores;
}

#endif // POSTPROCESS_HELPERS_H

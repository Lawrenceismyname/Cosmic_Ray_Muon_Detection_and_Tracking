#include "RunAction.hh"

#include <G4AnalysisManager.hh>
#include <G4Run.hh>
#include <string>

cRunAction::cRunAction() {
  // Get analysis manager instance
  auto analysisManager = G4AnalysisManager::Instance();

  // ROOT ntuple (optional)
  analysisManager->CreateNtuple("Hits", "Detector hits");
  analysisManager->CreateNtupleDColumn("Det_1_Photon_Energy_[eV]");
  analysisManager->CreateNtupleIColumn("Det_1_PhotonCount");
  analysisManager->CreateNtupleDColumn("Det_2_Photon_Energy_[eV]");
  analysisManager->CreateNtupleIColumn("Det_2_PhotonCount");
  analysisManager->CreateNtupleDColumn(
      "Muon_Offset_[m]");  // Add muon offset column
  analysisManager->CreateNtupleDColumn(
      "Det_1_FirstHitTime_[ns]");  // Add first hit time for SiPM1
  analysisManager->CreateNtupleDColumn(
      "Det_2_FirstHitTime_[ns]");  // Add first hit time for SiPM2

  analysisManager->FinishNtuple(0);
}

cRunAction::~cRunAction() {}

void cRunAction::BeginOfRunAction(const G4Run* aRun) {
  auto analysisManager = G4AnalysisManager::Instance();
  G4int runNumber = aRun->GetRunID();
  analysisManager->OpenFile("DetCount_" + std::to_string(runNumber) + ".csv");
}

void cRunAction::EndOfRunAction(const G4Run* aRun) {
  auto analysisManager = G4AnalysisManager::Instance();
  analysisManager->Write();
  analysisManager->CloseFile();
}

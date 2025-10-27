
#include "EventAction.hh"

#include <G4Event.hh>
#include <G4SystemOfUnits.hh>
#include <G4UserEventAction.hh>

#include "PrimaryGeneratorAction.hh"
#include "RunAction.hh"

cEventAction::cEventAction(cRunAction* runAction) : fRunAction(runAction) {
  fPrimaryGenerator = nullptr;  // Initialize to nullptr
  // Nothing else to do here
}

cEventAction::~cEventAction() {}

void cEventAction::BeginOfEventAction(const G4Event*) {
  edepSiPM[0] = edepSiPM[1] = 0.;
  photonCount[0] = photonCount[1] = 0;      // Initialize photon counts
  firstHitTime[0] = firstHitTime[1] = -1.;  // Initialize with invalid time
  fEdep = 0.;
}

void cEventAction::EndOfEventAction(const G4Event* event) {
  auto analysisManager = G4AnalysisManager::Instance();
  G4int eventID = event->GetEventID();

  // // Debug output
  // G4cout << "Event " << eventID << ": "
  //        << "SiPM1 photons=" << photonCount[0]
  //        << ", SiPM2 photons=" << photonCount[1] << G4endl;

  if (fPrimaryGenerator) {
    fMuOffset = fPrimaryGenerator->GetMuOffset();
  } else {
    fMuOffset = -1;  // Default value
  }

  // Fill ntuple
  analysisManager->FillNtupleDColumn(0, 0, edepSiPM[0]);
  analysisManager->FillNtupleIColumn(0, 1, photonCount[0]);
  analysisManager->FillNtupleDColumn(0, 2, edepSiPM[1]);
  analysisManager->FillNtupleIColumn(0, 3, photonCount[1]);
  analysisManager->FillNtupleDColumn(0, 4, fMuOffset);  // Add muon offset
  analysisManager->FillNtupleDColumn(0, 5,
                                     firstHitTime[0]);  // First hit time SiPM1
  analysisManager->FillNtupleDColumn(0, 6,
                                     firstHitTime[1]);  // First hit time SiPM2

  analysisManager->AddNtupleRow(0);  // important! writes one row per event
}

void cEventAction::AddPhotonCount(G4int detID) {
  if (detID == 1) {
    photonCount[detID - 1] += 1;
  }
  if (detID == 2) {
    photonCount[detID - 1] += 1;
  }
}

// Implement the methods
void cEventAction::AddEdep(G4int detID, G4double edep) {
  if (detID >= 1 && detID < 3) {
    edepSiPM[detID - 1] += edep;
  }
}

// New method to store first hit time
void cEventAction::AddPhotonTime(G4int detID, G4double hitTime) {
  if (detID >= 1 && detID < 3) {
    int index = detID - 1;
    // Store time only if it's the first hit or earlier than current first hit
    if (firstHitTime[index] < 0 || hitTime < firstHitTime[index]) {
      firstHitTime[index] = hitTime;
    }
  }
}

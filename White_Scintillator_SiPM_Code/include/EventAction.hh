#ifndef __EVENT_ACTION__HH
#define __EVENT_ACTION__HH
#include <G4Event.hh>
#include <G4EventManager.hh>
#include <G4Types.hh>
#include <G4UserEventAction.hh>

#include "RunAction.hh"

// Forward declaration
class cPrimaryGeneratorAction;

class cEventAction : public G4UserEventAction {
 public:
  cEventAction(cRunAction*);
  ~cEventAction();

  void BeginOfEventAction(const G4Event*) override;
  void EndOfEventAction(const G4Event*) override;

  inline void addEdep(G4double edep) { fEdep += edep; }

  void AddEdep(G4int detID, G4double edep);

  void AddPhotonCount(G4int detID);

  void AddPhotonTime(G4int detID,
                     G4double hitTime);  // Add method for time storage

  // Set the primary generator action
  void SetPrimaryGeneratorAction(cPrimaryGeneratorAction* primaryGenerator) {
    fPrimaryGenerator = primaryGenerator;
  }

  // Variables to store energy/time from SD
  G4double edepSiPM[2];
  G4int photonCount[2];      // Add photon counting
  G4double firstHitTime[2];  // Add first hit time storage for each detector
  G4double fMuOffset;

  cRunAction* fRunAction;  // pointer to RunAction for CSV output

 private:
  // Deposited energy
  G4double fEdep;
  cPrimaryGeneratorAction* fPrimaryGenerator;  // Pointer to primary generator
};

#endif
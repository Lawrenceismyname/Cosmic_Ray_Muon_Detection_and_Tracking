#include "SteppingAction.hh"

#include <G4RunManager.hh>
#include <G4Step.hh>
#include <G4Types.hh>

#include "DetectorConstruction.hh"

cSteppingAction::cSteppingAction(cEventAction* eventAction) {
  fEventAction = eventAction;
}

cSteppingAction::~cSteppingAction() {}

void cSteppingAction::UserSteppingAction(const G4Step* aStep) {
  ;
  G4LogicalVolume* volume = aStep->GetPreStepPoint()
                                ->GetTouchableHandle()
                                ->GetVolume()
                                ->GetLogicalVolume();

  const cDetectorConstruction* detectorConstruction =
      static_cast<const cDetectorConstruction*>(
          G4RunManager::GetRunManager()->GetUserDetectorConstruction());
  G4LogicalVolume* fScoringVolume = detectorConstruction->GetScoringVolume();

  if (volume == fScoringVolume) {
    G4double Edep = aStep->GetTotalEnergyDeposit();
    fEventAction->addEdep(Edep);
  }
}
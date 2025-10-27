#include "SensitiveDetector.hh"

#include <CLHEP/Units/SystemOfUnits.h>

#include <G4AnalysisManager.hh>
#include <G4OpticalPhoton.hh>
#include <G4RunManager.hh>
#include <G4StepPoint.hh>
#include <G4String.hh>
#include <G4SystemOfUnits.hh>
#include <G4ThreeVector.hh>
#include <G4TrackStatus.hh>
#include <G4Types.hh>
#include <G4VTouchable.hh>
#include <G4ios.hh>
#include <Randomize.hh>
#include <fstream>

#include "EventAction.hh"

cSensitiveDetector::cSensitiveDetector(const G4String& name)
    : G4VSensitiveDetector(name) {
  pdeCurve = new G4PhysicsOrderedFreeVector();
  std::ifstream dataFile;
  dataFile.open("eff.dat");

  while (1) {
    if (dataFile.eof()) break;
    G4double wlen, pde;
    dataFile >> wlen >> pde;

    pdeCurve->InsertValues(wlen * nm, pde / 100);
  }
  dataFile.close();
}
cSensitiveDetector::~cSensitiveDetector() {}

G4bool cSensitiveDetector::ProcessHits(G4Step* step, G4TouchableHistory*) {
  if (!step) return false;

  auto track = step->GetTrack();
  // G4double edep = step->GetTotalEnergyDeposit();
  // if (track->GetParentID() == 0) G4cout << "Edep: " << edep << "MeV" <<
  // G4endl; Only consider optical photons
  if (step->GetTrack()->GetDefinition() !=
      G4OpticalPhoton::OpticalPhotonDefinition())
    return false;

  auto preStep = step->GetPreStepPoint();
  G4int detID = preStep->GetTouchableHandle()->GetCopyNumber();

  if (detID < 1 || detID > 2) return false;  // safety check

  G4double photonEnergy = track->GetKineticEnergy();  // in MeV

  // Convert energy to wavelength for PDE lookup
  G4double wavelength = (h_Planck * c_light / photonEnergy);  // in nm

  // Get PDE from efficiency curve
  G4double pde = pdeCurve->Value(wavelength);

  // Apply detection efficiency - randomly reject based on PDE
  if (G4UniformRand() > pde) {
    // Photon not detected - kill it but don't count
    track->SetTrackStatus(fStopAndKill);
    return false;
  }
  // G4cout << " Photon wavelength: " << wavelength / nm << "nm" << G4endl;

  // Get the hit time from pre-step point
  G4double hitTime = preStep->GetGlobalTime();  // in ns

  // Access EventAction
  auto evtAction = static_cast<cEventAction*>(
      G4EventManager::GetEventManager()->GetUserEventAction());
  if (!evtAction) return false;

  // Count this photon
  // evtAction->AddEdep(detID, edep);

  evtAction->AddPhotonCount(detID);
  // evtAction->AddEdep(detID, photonEnergy / eV);
  evtAction->AddPhotonTime(detID, hitTime);  // Record the hit time

  // G4cout << "SiPM" << detID << " Photon energy: " << photonEnergy / eV << "
  // eV"
  //        << G4endl;

  // Kill photon
  step->GetTrack()->SetTrackStatus(fStopAndKill);

  return true;
}

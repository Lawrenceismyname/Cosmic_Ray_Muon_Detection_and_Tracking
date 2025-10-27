#include "PhysicsList.hh"

#include <G4DecayPhysics.hh>
#include <G4EmStandardPhysics.hh>
#include <G4OpticalPhysics.hh>
#include <G4Scintillation.hh>

cPhysicsList::cPhysicsList() {
  RegisterPhysics(new G4EmStandardPhysics);
  RegisterPhysics(new G4OpticalPhysics);
  RegisterPhysics(new G4DecayPhysics);
}

cPhysicsList::~cPhysicsList() {}
void cPhysicsList::ConstructProcess() {
  G4VModularPhysicsList::ConstructProcess();

  G4Scintillation* scintillationProcess = new G4Scintillation();
  scintillationProcess->SetFiniteRiseTime(true);
}

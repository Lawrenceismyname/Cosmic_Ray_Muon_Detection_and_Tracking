#include "PrimaryGeneratorAction.hh"

#include <CLHEP/Units/SystemOfUnits.h>

#include <G4Event.hh>
#include <G4ParticleDefinition.hh>
#include <G4ParticleGun.hh>
#include <G4ParticleTable.hh>
#include <G4String.hh>
#include <G4SystemOfUnits.hh>
#include <G4ThreeVector.hh>
#include <G4Types.hh>

#include "EventAction.hh"
#include "Randomize.hh"

cPrimaryGeneratorAction::cPrimaryGeneratorAction() {
  fParticleGun = new G4ParticleGun(1);
  G4ParticleTable* particleTable = G4ParticleTable::GetParticleTable();
  G4String particleName = "mu-";
  G4ParticleDefinition* particle = particleTable->FindParticle(particleName);

  
  fParticleGun->SetParticleDefinition(particle);

  
}

cPrimaryGeneratorAction::~cPrimaryGeneratorAction() { delete fParticleGun; }

void cPrimaryGeneratorAction::GeneratePrimaries(G4Event* anEvent) {
  // Generate random numbers in a specific range [min, max]
  G4double min_e = 1;
  G4double max_e = 10;
  G4double randomInRange_e = min_e + (max_e - min_e) * G4UniformRand();

  G4double energy = randomInRange_e * GeV;

  // Generate random numbers in a specific range [min, max]
  G4double min_y = -0.0045;
  G4double max_y = 0.0045;
  G4double randomInRange_y = min_y + (max_y - min_y) * G4UniformRand();

  G4float muOffset_y = randomInRange_y * m;

  // Generate random numbers in a specific range [min, max]
  G4double min_x = -0.1245;
  G4double max_x = 0.1245;
  G4double randomInRange_x = min_x + (max_x - min_x) * G4UniformRand();

  G4float muOffset_x = randomInRange_x * m;
  fMuOffset = randomInRange_x;  // Store the random offset

  G4ThreeVector position(muOffset_x, muOffset_y, 0.);
  G4ThreeVector momentumDirection(0, 0, 1);
  fParticleGun->SetParticlePosition(position);

  fParticleGun->SetParticleMomentumDirection(momentumDirection);
  fParticleGun->SetParticleEnergy(energy);


  fParticleGun->GeneratePrimaryVertex(anEvent);
}
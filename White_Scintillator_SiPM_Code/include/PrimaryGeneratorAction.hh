#ifndef __PRIMARY_GENERATOR_ACTION__HH
#define __PRIMARY_GENERATOR_ACTION__HH
#include <G4ParticleGun.hh>
#include <G4VUserPrimaryGeneratorAction.hh>

class cPrimaryGeneratorAction : public G4VUserPrimaryGeneratorAction {
 public:
  cPrimaryGeneratorAction();
  ~cPrimaryGeneratorAction();

  void GeneratePrimaries(G4Event *) override;

  G4double GetMuOffset() const { return fMuOffset; }

 private:
  G4ParticleGun *fParticleGun;
  G4double fMuOffset;
};
#endif
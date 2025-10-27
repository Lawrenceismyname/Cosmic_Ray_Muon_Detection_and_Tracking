#ifndef __SENSITIVE_DETECTOR__HH
#define __SENSITIVE_DETECTOR__HH
#include <G4PhysicsOrderedFreeVector.hh>
#include <G4String.hh>
#include <G4SystemOfUnits.hh>
#include <G4VSensitiveDetector.hh>


class cSensitiveDetector : public G4VSensitiveDetector {
 public:
  cSensitiveDetector(const G4String&);
  ~cSensitiveDetector();

 protected:
  G4bool ProcessHits(G4Step* aStep, G4TouchableHistory* ROhist) override;

 private:
  G4PhysicsOrderedFreeVector* pdeCurve;
  // for PDE curve ..
  static constexpr G4double h_Planck = 6.62607015e-34 * joule * second;
  static constexpr G4double c_light = 299792458.0 * meter / second;
};
#endif
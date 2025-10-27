#ifndef __DETECTOR_CONSTRUCTION__HH
#define __DETECTOR_CONSTRUCTION__HH
#include <sys/stat.h>

#include <G4Box.hh>
#include <G4Element.hh>
#include <G4GenericMessenger.hh>
#include <G4LogicalSkinSurface.hh>
#include <G4LogicalVolume.hh>
#include <G4Material.hh>
#include <G4MaterialPropertiesTable.hh>
#include <G4OpticalSurface.hh>
#include <G4SystemOfUnits.hh>
#include <G4Tubs.hh>
#include <G4Types.hh>
#include <G4VPhysicalVolume.hh>
#include <G4VUserDetectorConstruction.hh>

class cDetectorConstruction : public G4VUserDetectorConstruction {
 public:
  cDetectorConstruction();
  ~cDetectorConstruction();
  void DefineMaterials();
  void ConstructScintillators();
  G4VPhysicalVolume *Construct() override;
  inline G4LogicalVolume *GetScoringVolume() const { return fScoringVolume; }
  void ConstructSDandField() override;

 private:
  G4LogicalVolume *fScoringVolume;

  // Geometry dimensions
  // world Volume x dimension
  G4double worldWidth;
  // world Volume y dimension
  G4double worldLength;
  // world Volume z dimension
  G4double worldHeight;

  // Logical Volumes for SensitiveDetector, Radiator/Aetogel and World Volume
  G4LogicalVolume *logicWorld, *logicScint, *logicDet, *logicTeflon;
  // Solid Volumes for SensitiveDetector, Radiator/Aetogel and World Volume
  G4Box *solidDetector, *solidWorld, *solidRadiator, *solidScint;
  // Physical Volumes for SensitiveDetector, Radiator/Aetogel and World Volume
  G4VPhysicalVolume *physWorld, *physScint, *physDet1, *physDet2;
  // Materials
  G4Material *worldMat, *PVT, *fTeflon, *detMat;
  // Elements
  G4Element *elC, *elH;
  // Property tabless
  G4MaterialPropertiesTable *mptWorld, *mptPVT, *mptSi;

  // 200 nm , 900 nm
  static constexpr G4double startWaveLength = 380 * nm;
  static constexpr G4double stopWaveLength = 500 * nm;
  static constexpr G4double step = 5 * nm;
  static constexpr G4double eConst = (1.239841939 * 1000) * eV * nm;
  static constexpr G4int nEntries =
      ((stopWaveLength - startWaveLength) / step) + 1;
  G4double energy[nEntries];
  /**
   * Scintillator propersties (EJ200)
   */
  // Refractive index
  static constexpr G4double rIdxScint = 1.58;
  // Absorption Length
  static constexpr G4double absScint = 380 * cm;
  // Scintillation Efficiency (photons/1 MeV e-)
  static constexpr G4double effScint = 10000 / MeV;
  // Time constant
  static constexpr G4double decayTimeScint = 2.1 * ns;
  // Rise Time
  static constexpr G4double riseTimeScint = 0.9 * ns;
};
#endif
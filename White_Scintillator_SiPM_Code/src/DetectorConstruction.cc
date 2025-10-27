#include "DetectorConstruction.hh"

#include "SensitiveDetector.hh"

// Geant4 Headers
#include <G4Box.hh>
#include <G4Colour.hh>
#include <G4LogicalBorderSurface.hh>
#include <G4LogicalSkinSurface.hh>
#include <G4LogicalVolume.hh>
#include <G4LogicalVolumeStore.hh>
#include <G4Material.hh>
#include <G4MaterialPropertiesTable.hh>
#include <G4NistManager.hh>
#include <G4OpticalSurface.hh>
#include <G4PVPlacement.hh>
#include <G4PhysicalVolumeStore.hh>
#include <G4RunManager.hh>
#include <G4SolidStore.hh>
#include <G4SurfaceProperty.hh>
#include <G4ThreeVector.hh>
#include <G4Types.hh>
#include <G4VisAttributes.hh>
#include <G4ios.hh>

cDetectorConstruction::cDetectorConstruction()
    : worldWidth{0.5 * m}, worldLength{0.5 * m}, worldHeight{0.5 * m} {
  DefineMaterials();
}
void cDetectorConstruction::ConstructScintillators() {
  solidScint = new G4Box("solidScint", 125 * mm, 5 * mm, 5 * mm);
  logicScint = new G4LogicalVolume(solidScint, PVT, "logicScint");
  // Example: make turquoise
  G4Colour turquoise(0.25, 0.6, 0.7, 0.7);  // RGB values between 0–1

  // Create visualization attributes
  G4VisAttributes* turquoiseVis = new G4VisAttributes(turquoise);
  turquoiseVis->SetVisibility(true);
  turquoiseVis->SetForceSolid(true);  // If you want it solid (not wireframe)

  // Apply to logical volume
  logicScint->SetVisAttributes(turquoiseVis);
  fScoringVolume = logicScint;
  physScint = new G4PVPlacement(0, G4ThreeVector(0, 0, 0.1 * m), logicScint,
                                "physRDecay", logicWorld, false, 0, true);

  // --- Detector tiles
  // ----------------------------------------------------------------------------
  G4double detX = 0.1 * mm;
  G4double detY = 3 * mm;
  G4double detZ = 3 * mm;  // thickness in X-direction

  auto solidDet = new G4Box("solidDet", detX / 2, detY / 2, detZ / 2);

  logicDet = new G4LogicalVolume(solidDet, detMat, "logicDet");

  // Make the detector red 000

  G4Colour detect(0.5, 0, 0);  // RGB values between 0–1
  auto detVis = new G4VisAttributes(detect);
  detVis->SetVisibility(true);
  detVis->SetForceSolid(true);
  logicDet->SetVisAttributes(detVis);

  // Place detectors at each end of scintillator
  physDet1 = new G4PVPlacement(
      nullptr, G4ThreeVector(-125 * mm - detX / 2, 0, 0.1 * m),
      // G4ThreeVector(-125 * mm - detX / 2, 0, 0.25 * m),
      logicDet, "physDet", logicWorld, false, 1, true);

  physDet2 = new G4PVPlacement(
      nullptr, G4ThreeVector(+125 * mm + detX / 2, 0, 0.1 * m),
      // G4ThreeVector(-125 * mm - detX / 2, 0, 0.25 * m),
      logicDet, "physDet", logicWorld, false, 2, true);
}

cDetectorConstruction::~cDetectorConstruction() {}

G4VPhysicalVolume* cDetectorConstruction::Construct() {
  solidWorld =
      new G4Box("solidWorld", worldWidth / 2, worldLength / 2, worldHeight / 2);

  logicWorld = new G4LogicalVolume(solidWorld, worldMat, "logicWorld");

  physWorld = new G4PVPlacement(0, G4ThreeVector(0, 0, 0), logicWorld,
                                "physWorld", nullptr, false, 0, true);

  ConstructScintillators();
  G4double teflonEff[nEntries];
  G4double teflonRef[nEntries];
  for (G4int i = 0; i < nEntries; i++) {
    teflonEff[i] = 0.00;
    teflonRef[i] = 0.95;
  }
  G4double zScint = 0.1 * m;  // scintillator center
  G4double sx = 125 * mm, sy = 5 * mm, sz = 5 * mm;
  G4double thick = 1 * mm;

  // // +X face
  // auto* solidTefPlusX = new G4Box("TefPlusX", thick / 2, sy, sz);
  // auto* logicTefPlusX =
  //     new G4LogicalVolume(solidTefPlusX, fTeflon, "logicTefPlusX");
  // G4VPhysicalVolume* physTefPlusX = new G4PVPlacement(
  //     0, G4ThreeVector(sx + thick / 2, 0, zScint), logicTefPlusX,
  //     "physTefPlusX", logicWorld, false, 0, true);

  // -Y face
  auto solidTefMinusY = new G4Box("TefMinusY", sx, thick / 2, sz);
  auto logicTefMinusY =
      new G4LogicalVolume(solidTefMinusY, fTeflon, "logicTefMinusY");
  G4Colour reflect(0, 0.1, 0, 0.1);  // RGB values between 0–1
  auto detVis = new G4VisAttributes(reflect);
  detVis->SetVisibility(true);
  detVis->SetForceSolid(true);
  logicTefMinusY->SetVisAttributes(detVis);
  G4VPhysicalVolume* physTefMinusY = new G4PVPlacement(
      0, G4ThreeVector(0, -(sy + thick / 2), zScint), logicTefMinusY,
      "physTefMinusY", logicWorld, false, 0, true);

  // +Y face
  auto solidTefPlusY = new G4Box("TefPlusY", sx, thick / 2, sz);
  auto logicTefPlusY =
      new G4LogicalVolume(solidTefPlusY, fTeflon, "logicTefPlusY");
  logicTefPlusY->SetVisAttributes(detVis);
  G4VPhysicalVolume* physTefPlusY = new G4PVPlacement(
      0, G4ThreeVector(0, sy + thick / 2, zScint), logicTefPlusY,
      "physTefPlusY", logicWorld, false, 0, true);

  // -Z face
  auto solidTefMinusZ = new G4Box("TefMinusZ", sx, sy, thick / 2);
  auto logicTefMinusZ =
      new G4LogicalVolume(solidTefMinusZ, fTeflon, "logicTefMinusZ");
  logicTefMinusZ->SetVisAttributes(detVis);
  G4VPhysicalVolume* physTefMinusZ = new G4PVPlacement(
      0, G4ThreeVector(0, 0, zScint - (sz + thick / 2)), logicTefMinusZ,
      "physTefMinusZ", logicWorld, false, 0, true);

  // +Z face
  auto solidTefPlusZ = new G4Box("TefPlusZ", sx, sy, thick / 2);
  auto logicTefPlusZ =
      new G4LogicalVolume(solidTefPlusZ, fTeflon, "logicTefPlusZ");
  logicTefPlusZ->SetVisAttributes(detVis);
  G4VPhysicalVolume* physTefPlusZ = new G4PVPlacement(
      0, G4ThreeVector(0, 0, zScint + (sz + thick / 2)), logicTefPlusZ,
      "physTefPlusZ", logicWorld, false, 0, true);

  // --- Optical border surface between scintillator and each Teflon slab ---
  auto teflonSurface = new G4OpticalSurface("ScintTefSurface");
  teflonSurface->SetType(dielectric_metal);
  teflonSurface->SetModel(unified);
  teflonSurface->SetFinish(polishedteflonair);

  auto teflonMPT = new G4MaterialPropertiesTable();
  teflonMPT->AddProperty("REFLECTIVITY", energy, teflonRef, nEntries);
  teflonMPT->AddProperty("EFFICIENCY", energy, teflonEff, nEntries);
  teflonSurface->SetMaterialPropertiesTable(teflonMPT);

  // //Remove for 2 detectors
  // new G4LogicalBorderSurface("ScintPlusX_Border", physScint, physTefPlusX,
  //                            teflonSurface);

  new G4LogicalBorderSurface("ScintMinusY_Border", physScint, physTefMinusY,
                             teflonSurface);
  new G4LogicalBorderSurface("ScintPlusY_Border", physScint, physTefPlusY,
                             teflonSurface);
  new G4LogicalBorderSurface("ScintMinusZ_Border", physScint, physTefMinusZ,
                             teflonSurface);
  new G4LogicalBorderSurface("ScintPlusZ_Border", physScint, physTefPlusZ,
                             teflonSurface);

  // Note: -X face remains open for detector

  // -X face is left open for detector

  return physWorld;
}
void cDetectorConstruction::ConstructSDandField() {
  cSensitiveDetector* sensDet = new cSensitiveDetector("SensitiveDetector");
  logicDet->SetSensitiveDetector(sensDet);
  // if (isCherenkov) {
  //   logicDetector->SetSensitiveDetector(sensDet);
  // }
}

void cDetectorConstruction::DefineMaterials() {
  G4NistManager* nist = G4NistManager::Instance();
  worldMat = nist->FindOrBuildMaterial("G4_AIR");
  detMat = nist->FindOrBuildMaterial("G4_Si");

  elC = nist->FindOrBuildElement("C");
  elH = nist->FindOrBuildElement("H");

  PVT = new G4Material("PVT", 1.023 * g / cm3, 2);
  PVT->AddElement(elC, 9);
  PVT->AddElement(elH, 10);

  fTeflon = nist->FindOrBuildMaterial("G4_TEFLON");

  G4double emissionSpectScint[nEntries]{0,
                                        0,
                                        0,
                                        0,
                                        0.004651093470052131,
                                        0.09767434680826842,
                                        0.25116277336751297,
                                        0.546511679897461,
                                        0.867442079567871,
                                        0.9976746661783853,
                                        0.958139786171061,
                                        0.8744188262296548,
                                        0.7558140797509765,
                                        0.6465117863541666,
                                        0.5488373330891928,
                                        0.48604661313313796,
                                        0.437209386500651,
                                        0.3883721598681641,
                                        0.3209302399853516,
                                        0.23720928004394531,
                                        0.17674410682291686,
                                        0.13953482677897155,
                                        0.10465109347005225,
                                        0.0813953067496746,
                                        0.05581386683756509

  };
  G4double rIndexAir[nEntries];
  G4double rIndexSi[nEntries];
  G4double rIndexPVT[nEntries];
  G4double absPVT[nEntries];  // scintillator absorption length
  G4double teflonEff[nEntries];
  G4double teflonRef[nEntries];
  for (G4int i = 0; i < nEntries; i++) {
    G4double wavelength = startWaveLength + step * (nEntries - i - 1);

    energy[i] = eConst / wavelength;
    teflonEff[i] = 0.03;
    teflonRef[i] = 0.97;
    rIndexAir[i] = 1;
    rIndexSi[i] = 1.59;
    rIndexPVT[i] = rIdxScint;
    absPVT[i] = absScint;
  }
  mptPVT = new G4MaterialPropertiesTable;
  mptWorld = new G4MaterialPropertiesTable;
  mptSi = new G4MaterialPropertiesTable;
  mptWorld->AddProperty("RINDEX", energy, rIndexAir, nEntries);
  mptSi->AddProperty("RINDEX", energy, rIndexSi, nEntries);

  // // PVT Scintillator
  mptPVT->AddProperty("RINDEX", energy, rIndexPVT, nEntries);
  mptPVT->AddProperty("ABSLENGTH", energy, absPVT, nEntries);
  mptPVT->AddProperty("SCINTILLATIONCOMPONENT1", energy, emissionSpectScint,
                      nEntries);
  mptPVT->AddConstProperty("SCINTILLATIONYIELD", effScint);
  mptPVT->AddConstProperty("SCINTILLATIONTIMECONSTANT1", decayTimeScint);
  mptPVT->AddConstProperty("SCINTILLATIONRISETIME1", riseTimeScint);
  mptPVT->AddConstProperty("RESOLUTIONSCALE", 1.0);

  PVT->SetMaterialPropertiesTable(mptPVT);
  worldMat->SetMaterialPropertiesTable(mptWorld);
  detMat->SetMaterialPropertiesTable(mptSi);
  // Set the Birks Constant for the LXe scintillator
  PVT->GetIonisation()->SetBirksConstant(0.126 * mm / MeV);
}
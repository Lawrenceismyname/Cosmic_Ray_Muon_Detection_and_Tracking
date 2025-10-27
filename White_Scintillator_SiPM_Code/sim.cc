#include <FTFP_BERT.hh>
#include <G4OpticalPhysics.hh>
#include <G4RunManagerFactory.hh>
#include <G4String.hh>
#include <G4UIExecutive.hh>
#include <G4UImanager.hh>
#include <G4VisExecutive.hh>
#include <G4ios.hh>

#include "G4EmStandardPhysics_option4.hh"

// Local Includes
#include "ActionInitialization.hh"
#include "DetectorConstruction.hh"

int main(int argc, char* argv[]) {
  G4UIExecutive* ui = nullptr;

  if (argc == 1) {
    ui = new G4UIExecutive(argc, argv);
  }
  auto runManager =
      G4RunManagerFactory::CreateRunManager(G4RunManagerType::Default);
  runManager->SetUserInitialization(new cDetectorConstruction);
  // runManager->SetUserInitialization(new cPhysicsList);
  G4VModularPhysicsList* physicsList = new FTFP_BERT;
  physicsList->ReplacePhysics(new G4EmStandardPhysics_option4());

  auto opticalPhysics = new G4OpticalPhysics();
  auto opticalParams = G4OpticalParameters::Instance();

  // opticalParams->SetWLSTimeProfile("delta");

  opticalParams->SetScintTrackSecondariesFirst(true);

  // opticalParams->SetCerenkovMaxPhotonsPerStep(100);
  // opticalParams->SetCerenkovMaxBetaChange(10.0);
  // opticalParams->SetCerenkovTrackSecondariesFirst(true);

  physicsList->RegisterPhysics(opticalPhysics);
  runManager->SetUserInitialization(physicsList);
  runManager->SetUserInitialization(new cActionInitialization);
  ;
  G4VisManager* visManager = new G4VisExecutive(argc, argv);
  visManager->Initialize();
  G4UImanager* uiManager = G4UImanager::GetUIpointer();
  if (ui != nullptr) {
    uiManager->ApplyCommand("/control/execute vis.mac");
    ui->SessionStart();
    delete ui;
  }

  else {
    G4String command = "/control/execute ";
    G4String fileName = argv[1];
    uiManager->ApplyCommand(command + fileName);
  }
  delete visManager;
  delete runManager;

  return 0;
}
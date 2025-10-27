#ifndef __RUN_ACTION__HH
#define __RUN_ACTION__HH
#include <G4AnalysisManager.hh>
#include <G4UserRunAction.hh>


class cRunAction : public G4UserRunAction {
 public:
  cRunAction();
  ~cRunAction();
  void BeginOfRunAction(const G4Run* /*aRun*/) override;
  void EndOfRunAction(const G4Run* /*aRun*/) override;



 private:
  G4AnalysisManager* analysisManager;
  
};
#endif
#ifndef __STEPPING_ACTION__HH
#define __STEPPING_ACTION__HH
#include <G4Step.hh>
#include <G4UserSteppingAction.hh>

#include "EventAction.hh"


class cSteppingAction : public G4UserSteppingAction {
 public:
  cSteppingAction(cEventAction* eventAction);
  ~cSteppingAction();
  void UserSteppingAction(const G4Step*) override;

 private:
  cEventAction* fEventAction;
};

#endif
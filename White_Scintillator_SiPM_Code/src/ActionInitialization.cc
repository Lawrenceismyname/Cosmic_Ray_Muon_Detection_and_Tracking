#include "ActionInitialization.hh"

#include "EventAction.hh"
#include "PrimaryGeneratorAction.hh"
#include "RunAction.hh"

cActionInitialization::cActionInitialization() {}
cActionInitialization::~cActionInitialization() {};
void cActionInitialization::Build() const {
  cPrimaryGeneratorAction *primaryGenerator = new cPrimaryGeneratorAction();
  SetUserAction(primaryGenerator);

  cRunAction *runAction = new cRunAction();
  SetUserAction(runAction);

  cEventAction *eventAction = new cEventAction(runAction);
  eventAction->SetPrimaryGeneratorAction(primaryGenerator);  // THIS IS MISSING!

  SetUserAction(eventAction);
}
void cActionInitialization::BuildForMaster() const {
  cRunAction *runAction = new cRunAction();
  SetUserAction(runAction);
}
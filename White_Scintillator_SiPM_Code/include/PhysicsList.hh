#ifndef __PHYSICS_LIST__INC
#define __PHYSICS_LIST__INC
#include <G4VModularPhysicsList.hh>
class cPhysicsList : public G4VModularPhysicsList {
 public:
  cPhysicsList();
  ~cPhysicsList();
  void ConstructProcess() override;
};
#endif
#ifndef __ACTION_INITIALIZATION__HH
#define __ACTION_INITIALIZATION__HH
#include <G4VUserActionInitialization.hh>
class cActionInitialization : public G4VUserActionInitialization {
 public:
  cActionInitialization();
  ~cActionInitialization() override;

  void Build() const override;
  void BuildForMaster() const override;
};
#endif
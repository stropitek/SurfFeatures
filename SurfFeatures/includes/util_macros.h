#ifndef UTIL_MACROS
#define UTIL_MACROS

#define GETSET(Type, MemberName, FaceName) \
            Type get##FaceName() const { \
                return MemberName; \
          }; \
          void set##FaceName(Type value) { \
                MemberName = value; \
          }
#define GET(Type, MemberName, FaceName) \
          Type get##FaceName() const { \
             return MemberName; \
          }
#define SET(Type, MemberName, FaceName) \
            void set##FaceName(const Type &value) { \
                MemberName = value; \
            }

#define SLOTDEF_1(Type, FuncName,LogicFuncName)\
  void qSlicerSurfFeaturesModuleWidget::FuncName(Type value){\
    Q_D(qSlicerSurfFeaturesModuleWidget);\
    vtkSlicerSurfFeaturesLogic* logic = d->logic();\
    logic->LogicFuncName(value);\
  }

#define SLOTDEF_0(FuncName,LogicFuncName)\
  void qSlicerSurfFeaturesModuleWidget::FuncName(){\
    Q_D(qSlicerSurfFeaturesModuleWidget);\
    vtkSlicerSurfFeaturesLogic* logic = d->logic();\
    logic->LogicFuncName();\
  }

#endif
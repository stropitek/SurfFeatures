/*==============================================================================

  Program: 3D Slicer

  Portions (c) Copyright Brigham and Women's Hospital (BWH) All Rights Reserved.

  See COPYRIGHT.txt
  or http://www.slicer.org/copyright/copyright.txt for details.

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.

==============================================================================*/

// Qt includes
#include <QtPlugin>

// SurfFeatures Logic includes
#include <vtkSlicerSurfFeaturesLogic.h>

// SurfFeatures includes
#include "qSlicerSurfFeaturesModule.h"
#include "qSlicerSurfFeaturesModuleWidget.h"

//-----------------------------------------------------------------------------
Q_EXPORT_PLUGIN2(qSlicerSurfFeaturesModule, qSlicerSurfFeaturesModule);

//-----------------------------------------------------------------------------
/// \ingroup Slicer_QtModules_ExtensionTemplate
class qSlicerSurfFeaturesModulePrivate
{
public:
  qSlicerSurfFeaturesModulePrivate();
};

//-----------------------------------------------------------------------------
// qSlicerSurfFeaturesModulePrivate methods

//-----------------------------------------------------------------------------
qSlicerSurfFeaturesModulePrivate
::qSlicerSurfFeaturesModulePrivate()
{
}

//-----------------------------------------------------------------------------
// qSlicerSurfFeaturesModule methods

//-----------------------------------------------------------------------------
qSlicerSurfFeaturesModule
::qSlicerSurfFeaturesModule(QObject* _parent)
  : Superclass(_parent)
  , d_ptr(new qSlicerSurfFeaturesModulePrivate)
{
}

//-----------------------------------------------------------------------------
qSlicerSurfFeaturesModule::~qSlicerSurfFeaturesModule()
{
}

//-----------------------------------------------------------------------------
QString qSlicerSurfFeaturesModule::helpText()const
{
  return "This is a loadable module bundled in an extension";
}

//-----------------------------------------------------------------------------
QString qSlicerSurfFeaturesModule::acknowledgementText()const
{
  return "This work was was partially funded by NIH grant 3P41RR013218-12S1";
}

//-----------------------------------------------------------------------------
QStringList qSlicerSurfFeaturesModule::contributors()const
{
  QStringList moduleContributors;
  moduleContributors << QString("Jean-Christophe Fillion-Robin (Kitware)");
  return moduleContributors;
}

//-----------------------------------------------------------------------------
QIcon qSlicerSurfFeaturesModule::icon()const
{
  return QIcon(":/Icons/SurfFeatures.png");
}

//-----------------------------------------------------------------------------
QStringList qSlicerSurfFeaturesModule::categories() const
{
  return QStringList() << "Examples";
}

//-----------------------------------------------------------------------------
QStringList qSlicerSurfFeaturesModule::dependencies() const
{
  return QStringList();
}

//-----------------------------------------------------------------------------
void qSlicerSurfFeaturesModule::setup()
{
  this->Superclass::setup();
}

//-----------------------------------------------------------------------------
qSlicerAbstractModuleRepresentation * qSlicerSurfFeaturesModule
::createWidgetRepresentation()
{
  return new qSlicerSurfFeaturesModuleWidget;
}

//-----------------------------------------------------------------------------
vtkMRMLAbstractLogic* qSlicerSurfFeaturesModule::createLogic()
{
  return vtkSlicerSurfFeaturesLogic::New();
}

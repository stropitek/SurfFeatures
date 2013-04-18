/*==============================================================================

  Program: 3D Slicer

  Copyright (c) Kitware Inc.

  See COPYRIGHT.txt
  or http://www.slicer.org/copyright/copyright.txt for details.

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.

  This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc.
  and was partially funded by NIH grant 3P41RR013218-12S1

==============================================================================*/

#ifndef __qSlicerSurfFeaturesFooBarWidget_h
#define __qSlicerSurfFeaturesFooBarWidget_h

// Qt includes
#include <QWidget>

// FooBar Widgets includes
#include "qSlicerSurfFeaturesModuleWidgetsExport.h"

class qSlicerSurfFeaturesFooBarWidgetPrivate;

/// \ingroup Slicer_QtModules_SurfFeatures
class Q_SLICER_MODULE_SURFFEATURES_WIDGETS_EXPORT qSlicerSurfFeaturesFooBarWidget
  : public QWidget
{
  Q_OBJECT
public:
  typedef QWidget Superclass;
  qSlicerSurfFeaturesFooBarWidget(QWidget *parent=0);
  virtual ~qSlicerSurfFeaturesFooBarWidget();

protected slots:
void onFooBar();

protected:
  QScopedPointer<qSlicerSurfFeaturesFooBarWidgetPrivate> d_ptr;

private:
  Q_DECLARE_PRIVATE(qSlicerSurfFeaturesFooBarWidget);
  Q_DISABLE_COPY(qSlicerSurfFeaturesFooBarWidget);
};

#endif

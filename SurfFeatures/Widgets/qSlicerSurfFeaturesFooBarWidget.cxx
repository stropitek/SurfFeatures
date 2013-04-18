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

// FooBar Widgets includes
#include "qSlicerSurfFeaturesFooBarWidget.h"
#include "ui_qSlicerSurfFeaturesFooBarWidget.h"

#include <iostream>
#include <fstream>

#include <cv.h>

//-----------------------------------------------------------------------------
/// \ingroup Slicer_QtModules_SurfFeatures
class qSlicerSurfFeaturesFooBarWidgetPrivate
  : public Ui_qSlicerSurfFeaturesFooBarWidget
{
  Q_DECLARE_PUBLIC(qSlicerSurfFeaturesFooBarWidget);
protected:
  qSlicerSurfFeaturesFooBarWidget* const q_ptr;

public:
  qSlicerSurfFeaturesFooBarWidgetPrivate(
    qSlicerSurfFeaturesFooBarWidget& object);
  virtual void setupUi(qSlicerSurfFeaturesFooBarWidget*);
};

// --------------------------------------------------------------------------
qSlicerSurfFeaturesFooBarWidgetPrivate
::qSlicerSurfFeaturesFooBarWidgetPrivate(
  qSlicerSurfFeaturesFooBarWidget& object)
  : q_ptr(&object)
{
}

// --------------------------------------------------------------------------
void qSlicerSurfFeaturesFooBarWidgetPrivate
::setupUi(qSlicerSurfFeaturesFooBarWidget* widget)
{
  this->Ui_qSlicerSurfFeaturesFooBarWidget::setupUi(widget);
}

//-----------------------------------------------------------------------------
// qSlicerSurfFeaturesFooBarWidget methods

//-----------------------------------------------------------------------------
qSlicerSurfFeaturesFooBarWidget
::qSlicerSurfFeaturesFooBarWidget(QWidget* parentWidget)
  : Superclass( parentWidget )
  , d_ptr( new qSlicerSurfFeaturesFooBarWidgetPrivate(*this) )
{
  Q_D(qSlicerSurfFeaturesFooBarWidget);
  d->setupUi(this);

  connect(d->FooBarButton, SIGNAL(clicked()), this, SLOT(onFooBar()));
}

//-----------------------------------------------------------------------------
qSlicerSurfFeaturesFooBarWidget
::~qSlicerSurfFeaturesFooBarWidget()
{
}

void qSlicerSurfFeaturesFooBarWidget
::onFooBar()
{
	cv::Complexf cf;

}

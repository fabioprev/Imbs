#include "VisualReading.h"

using namespace std;
using namespace cv;

VisualReading::VisualReading() {;}

VisualReading::~VisualReading() {;}

void VisualReading::setObservations(const vector<Observation>& obs)
{
	observations = obs;
}

void VisualReading::setObservationsAgentPose(const Point2f& pose)
{
	observationsAgentPose = pose;
}

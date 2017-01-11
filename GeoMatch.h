#include <opencv2\core.hpp>
#include <opencv2\opencv.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <math.h>

using namespace cv;

class GeoMatch
{
private:
	int				noOfCordinates;		//Number of elements in coordinate array
	Point			*cordinates;		//Coordinates array to store model points	
	int				modelHeight;		//Template height
	int				modelWidth;			//Template width
	double			*edgeMagnitude;		//gradient magnitude
	double			*edgeDerivativeX;	//gradient in X direction
	double			*edgeDerivativeY;	//radient in Y direction	
	Point			centerOfGravity;	//Center of gravity of template 

	bool			modelDefined;

	void CreateDoubleMatrix(double **&matrix, CvSize size);
	void ReleaseDoubleMatrix(double **&matrix, int size);
public:
	GeoMatch(void);
	GeoMatch(const void* templateArr);
	~GeoMatch(void);

	int CreateGeoMatchModel(Mat templateArr, double, double);
	double FindGeoMatchModel(Mat srcarr, double minScore, double greediness, Point *resultPoint);
	void DrawContours(Mat pImage, Point COG, Scalar, int);
	void DrawContours(Mat pImage, Scalar, int);
};

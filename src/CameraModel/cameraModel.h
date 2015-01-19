/***************************************************************************
 *   cameraModel.h     - description
 *
 *   This program is part of the ETISEO project.
 *
 *   See http://www.etiseo.net  http://www.silogic.fr    
 *
 *   (C) Silogic - ETISEO Consortium
 ***************************************************************************/



#ifndef _CAMERA_MODEL_OBJECT_H_
#define _CAMERA_MODEL_OBJECT_H_

#include "xmlUtil.h"

#include <math.h>

#include <libxml/xmlwriter.h>
#include <libxml/xmlreader.h>


#define XML_TAG_CAMERA			BAD_CAST"Camera"
#define XML_TAG_NAME			BAD_CAST"name"


#define XML_TAG_GEOMETRY		BAD_CAST"Geometry"
#define XML_TAG_WIDTH			BAD_CAST"width"
#define XML_TAG_HEIGHT			BAD_CAST"height"
#define XML_TAG_NCX				BAD_CAST"ncx"
#define XML_TAG_NFX				BAD_CAST"nfx"
#define XML_TAG_DX				BAD_CAST"dx"
#define XML_TAG_DY				BAD_CAST"dy"
#define XML_TAG_DPX				BAD_CAST"dpx"
#define XML_TAG_DPY				BAD_CAST"dpy"

#define XML_TAG_INTRINSIC		BAD_CAST"Intrinsic"
#define XML_TAG_FOCAL			BAD_CAST"focal" 
#define XML_TAG_KAPPA1			BAD_CAST"kappa1" 
#define XML_TAG_CX				BAD_CAST"cx" 
#define XML_TAG_CY				BAD_CAST"cy" 
#define XML_TAG_SX				BAD_CAST"sx" 

#define XML_TAG_EXTRINSIC		BAD_CAST"Extrinsic"
#define XML_TAG_TX				BAD_CAST"tx" 
#define XML_TAG_TY				BAD_CAST"ty" 
#define XML_TAG_TZ				BAD_CAST"tz" 
#define XML_TAG_RX				BAD_CAST"rx" 
#define XML_TAG_RY				BAD_CAST"ry" 
#define XML_TAG_RZ				BAD_CAST"rz" 



namespace Etiseo {
	
 //!  A root class handling camera model
 class CameraModel
 {
    public:
      
	  //! Constructor
      CameraModel();
      //! Destructor
      virtual ~CameraModel();
	  
	  //! Access to members
	  std::string name() { return mName; }
	  const std::string& name() const { return mName; }
	  void setName(const std::string& name) { mName = name; }
	  
	  inline int width() const { return mImgWidth; }
	  inline int height() const { return mImgHeight; }
	  inline float ncx() const { return mNcx; }
	  inline float nfx() const { return mNfx; }
	  inline float dx() const { return mDx; }
	  inline float dy() const { return mDy; }
	  inline float dpx() const { return mDpx; }
	  inline float dpy() const { return mDpy; }
	  inline float cx() const { return mCx; }
	  inline float cy() const { return mCy; }
	  inline float sx() const { return mSx; }
	  inline float focal() const { return mFocal; }
	  inline float kappa1() const { return mKappa1; }
	  inline float tx() const { return mTx; }
	  inline float ty() const { return mTy; }
	  inline float tz() const { return mTz; }
	  inline float rx() const { return mRx; }
	  inline float ry() const { return mRy; }
	  inline float rz() const { return mRz; }
	  inline float cposx() const { return mCposx; }
	  inline float cposy() const { return mCposy; }
	  inline float cposz() const { return mCposz; }
	  
	  void setGeometry(int width, int height, float ncx, float nfx, 
	  					float dx, float dy, float dpx, float dpy);
		
	  void setIntrinsic(float focal, float kappa1, float cx, float cy, float sx);
	  
	  void setExtrinsic(float tx, float ty, float tz, float rx, float ry, float rz);
	   
	  //! Loading from an XML
	  virtual bool fromXml(std::istream& is);
	  //! Saving to an XML
	  virtual void toXml(std::ostream& os) const;
	  
	  //! Loading from a .dat files = output of the Tsai calibration 
	  virtual bool fromDat(std::istream& is, int width, int height);
	  
	  //! Coordinate manipulation
	  //! from image coordinate to world coordinate
	  bool imageToWorld(float Xi, float Yi, float Zw, float& Xw, float &Yw);
	  
	  //! from world coordinate to image coordinate
	  bool worldToImage(float Xw, float Yw, float Zw, float& Xi, float& Yi);
	  
	  //! convert from undistorted to distorted image
	  bool undistortedToDistortedImageCoord (float Xfu, float Yfu, float& Xfd, float& Yfd);
	  //! convert from distorted to undistorted image
	  bool distortedToUndistortedImageCoord (float Xfd, float Yfd, float& Xfu, float& Yfu);
	  
	  //! from world coordinate to camera coordinate
	  bool worldToCameraCoord (float xw, float yw, float zw, float& xc, float& yc, float& zc);
	  //! from camera coordinate to world coordinate
	  bool cameraToWorldCoord (float xc, float yc, float zc, float& xw, float& yw, float& zw);

	protected:
	
		virtual void internalInit();
		
		//! Coordinate manipulation, intermediate transformation :
		//! convert from distorted to undistorted sensor plane coordinates 
		void distortedToUndistortedSensorCoord (float Xd, float Yd, float& Xu, float& Yu);
		//! convert from undistorted to distorted sensor plane coordinates
		void undistortedToDistortedSensorCoord (float Xu, float Yu, float& Xd, float& Yd);
		
	private:
		
		bool			isInit;
		std::string		mName;
		
		// geometry
		int				mImgWidth;
		int				mImgHeight;
		float			mNcx;
		float			mNfx;
		float			mDx;
		float			mDy;
		float			mDpx;
		float			mDpy;

		// intrinsic 
		float			mFocal;
		float			mKappa1;
		float			mCx;
		float			mCy;
		float			mSx;
 
		// extrinsic 
		float			mTx;
		float			mTy;
		float			mTz;
		float			mRx;
		float			mRy;
		float			mRz;
		
		// for computation
		float			mR11;
		float			mR12;
		float			mR13;
		float			mR21;
		float			mR22;
		float			mR23;
		float			mR31;
		float			mR32;
		float			mR33;
		
		//camera position
		float			mCposx;
		float			mCposy;
		float			mCposz;
		
 };
};

#endif

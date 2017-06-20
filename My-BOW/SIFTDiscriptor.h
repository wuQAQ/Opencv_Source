#ifndef _SIFT_DISCRIPTOR_H_  
#define _SIFT_DISCRIPTOR_H_  
#include <string>  
#include <highgui.h>  
#include <cv.h>  
  
extern "C"  
{     
#include "../sift/sift.h"     
#include "../sift/imgfeatures.h"      
#include "../sift/utils.h"    
};  
  
class CSIFTDiscriptor  
{     
public:   
    int GetInterestPointNumber()          
    {         
        return m_nInterestPointNumber;    
    }     
    struct feature *GetFeatureArray()         
    {         
        return m_pFeatureArray;       
    }  
    public :          
        void SetImgName(const std::string &strImgName)        
        {         
            m_strInputImgName = strImgName;       
        }     
        int CalculateSIFT();  
    public:   
        CSIFTDiscriptor(const std::string &strImgName);   
        CSIFTDiscriptor()         
        {         
            m_nInterestPointNumber = 0;  
            m_pFeatureArray = NULL;       
        }     
        ~CSIFTDiscriptor();  
    private:          
        std::string m_strInputImgName;    
        int m_nInterestPointNumber;   
        feature *m_pFeatureArray;     
};  
#endif  
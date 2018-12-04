#ifndef SWIFTPR_PIPLINE_H
#define SWIFTPR_PIPLINE_H

#include "plateDetection.h"
#include "PlateInfo.h"
#include "fastDeskew.h"
#include "fineTuning.h"
#include "segmentationFreeRecognizer.h"

namespace pr{
    class PipelinePR{
        public:
            PlateDetection *plateDetection;
            FineMapping *fineMapping;
			PipelinePR() {};
			void initial(const std::string &detector_filename,
                       const std::string& finemapping_prototxt,const std::string& finemapping_caffemodel,
                       const std::string& segmentationfree_proto,const std::string& segmentationfree_caffemodel
                       )
			{
				plateDetection = new PlateDetection(detector_filename);
				fineMapping = new FineMapping(finemapping_prototxt, finemapping_caffemodel);

			};
			~PipelinePR() {};
		
            std::vector<std::string> plateRes;
     };


}
#endif //SWIFTPR_PIPLINE_H

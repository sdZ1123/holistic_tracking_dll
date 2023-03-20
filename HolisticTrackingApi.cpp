#include "HolisticTrackingApi.h"
#include "HolisticTrackingDetect.h"

using namespace GoogleMediapipeDetect;

HolisticTrackingDetect m_HolisticTrackingDetect;

EXPORT_API int MediapipeHolisticTrackingInit(
	const char* model_path,
	bool is_need_video_outputstream,
	bool is_need_pose_outputstream,
	bool is_need_hand_outputstream,
	bool is_need_face_outputstream,
	int modelComplexity,
	bool smoothLandmarks,
	bool usePrevLandmarks,
	bool refineFaceLandmarks)
{
	return m_HolisticTrackingDetect.InitModel(
		model_path,
		is_need_video_outputstream,
		is_need_pose_outputstream,
		is_need_hand_outputstream,
		is_need_face_outputstream,
		modelComplexity,
		smoothLandmarks,
		usePrevLandmarks,
		refineFaceLandmarks);
}

EXPORT_API int MediapipeHolisticTrackingDetectFrameDirect(
	int image_width,
	int image_height,
	void* image_data,
	std::vector<cv::Point3f>& pose,
	std::vector<cv::Point2f>& leftHandPose,
	std::vector<cv::Point2f>& rightHandPose,
	std::vector<cv::Point3f>& facePose,
	bool show_result_image)
{
	return m_HolisticTrackingDetect.DetectImageDirect(
		image_width,
		image_height,
		image_data,
		pose,
		leftHandPose,
		rightHandPose,
		facePose,
		show_result_image);
}


EXPORT_API int MediapipeHolisticTrackingRelease()
{
	return m_HolisticTrackingDetect.Release();
}
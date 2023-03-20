#include <vector>

#include "HolisticTrackingDetect.h"


GoogleMediapipeDetect::HolisticTrackingDetect::HolisticTrackingDetect()
{
	m_bIsInit = false;
	m_bIsRelease = false;

	m_Video_InputStreamName = "image";
	m_Video_OutputStreamName = "output_video";
	m_PoseLandmarks_OutputStreamName = "pose_landmarks";
	m_LeftHandLandmarks_OutputStreamName = "left_hand_landmarks";
	m_RightHandLandmarks_OutputStreamName = "right_hand_landmarks";
	m_FaceLandmarks_OutputStreamName = "face_landmarks";

	m_pVideoPoller = nullptr;
	m_pPoseLandmarksPoller = nullptr;
	m_pLeftHandLandmarksPoller = nullptr;
	m_pRightHandLandmarksPoller = nullptr;
	m_pFaceLandmarksPoller = nullptr;
}

GoogleMediapipeDetect::HolisticTrackingDetect::~HolisticTrackingDetect()
{
	if (m_bIsInit && !m_bIsRelease)
	{
		Release();
	}
}

int GoogleMediapipeDetect::HolisticTrackingDetect::InitModel(
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
	absl::Status run_status = Mediapipe_InitGraph(
		model_path,
		is_need_video_outputstream,
		is_need_pose_outputstream,
		is_need_hand_outputstream,
		is_need_face_outputstream,
		modelComplexity,
		smoothLandmarks,
		usePrevLandmarks,
		refineFaceLandmarks);
	if (!run_status.ok())
		return 0;
	m_bIsInit = true;
	return  1;
}

int GoogleMediapipeDetect::HolisticTrackingDetect::DetectImageDirect(
	int image_width,
	int image_height,
	void* image_data,
	std::vector<cv::Point3f>& pose,
	std::vector<cv::Point2f>& leftHandPose,
	std::vector<cv::Point2f>& rightHandPose,
	std::vector<cv::Point3f>& facePose,
	bool show_result_image)
{
	/*if (!m_bIsInit)
		return 0;*/

	absl::Status run_status = Mediapipe_RunMPPGraph_Direct(
		image_width,
		image_height,
		image_data,
		pose,
		leftHandPose,
		rightHandPose,
		facePose,
		show_result_image);
	if (!run_status.ok()) {
		return 0;
	}
	return 1;
}


int GoogleMediapipeDetect::HolisticTrackingDetect::Release()
{
	absl::Status run_status = Mediapipe_ReleaseGraph();
	if (!run_status.ok()) {
		return 0;
	}
	m_bIsRelease = true;
	return 1;
}

absl::Status GoogleMediapipeDetect::HolisticTrackingDetect::Mediapipe_InitGraph(
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
	std::string calculator_graph_config_contents;
	MP_RETURN_IF_ERROR(mediapipe::file::GetContents(model_path, &calculator_graph_config_contents));
	std::cout << "mediapipe::file::GetContents success" << std::endl;

	mediapipe::CalculatorGraphConfig config =
		mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(
			calculator_graph_config_contents);

	std::map<std::string, mediapipe::Packet> input_side_packets;
	/*int modelComplexity = 0;
	bool smoothLandmarks = false;
	bool refineFaceLandmarks = false;
	bool usePrevLandmarks = false;*/
	input_side_packets["model_complexity"] = mediapipe::MakePacket<int>(modelComplexity);
	input_side_packets["smooth_landmarks"] = mediapipe::MakePacket<bool>(smoothLandmarks);
	input_side_packets["refine_face_landmarks"] = mediapipe::MakePacket<bool>(refineFaceLandmarks);
	input_side_packets["use_prev_landmarks"] = mediapipe::MakePacket<bool>(usePrevLandmarks);


	MP_RETURN_IF_ERROR(m_Graph.Initialize(config, input_side_packets));
	std::cout << "m_Graph.Initialize(config) success" << std::endl;

	// 1 视频输出
	if(is_need_video_outputstream)
	{
		auto videoOutputStream = m_Graph.AddOutputStreamPoller(m_Video_OutputStreamName);
		assert(videoOutputStream.ok());
		m_pVideoPoller = std::make_unique<mediapipe::OutputStreamPoller>(std::move(videoOutputStream.value()));
	}

	// 2 PoseLandmarks输出
	if(is_need_pose_outputstream)
	{
		mediapipe::StatusOrPoller poseLandmarks = m_Graph.AddOutputStreamPoller(m_PoseLandmarks_OutputStreamName);
		assert(poseLandmarks.ok());
		m_pPoseLandmarksPoller = std::make_unique<mediapipe::OutputStreamPoller>(std::move(poseLandmarks.value()));
	}

	// 3 LeftHandLandmarks输出
	if(is_need_hand_outputstream)
	{
		mediapipe::StatusOrPoller leftHandLandmarks = m_Graph.AddOutputStreamPoller(m_LeftHandLandmarks_OutputStreamName);
		assert(leftHandLandmarks.ok());
		m_pLeftHandLandmarksPoller = std::make_unique<mediapipe::OutputStreamPoller>(std::move(leftHandLandmarks.value()));
	}

	// 4 RightHandLandmarks输出
	if(is_need_hand_outputstream)
	{
		mediapipe::StatusOrPoller rightHandLandmarks = m_Graph.AddOutputStreamPoller(m_RightHandLandmarks_OutputStreamName);
		assert(rightHandLandmarks.ok());
		m_pRightHandLandmarksPoller = std::make_unique<mediapipe::OutputStreamPoller>(std::move(rightHandLandmarks.value()));
	}

	// 5 FaceLandmarks输出
	if(is_need_face_outputstream)
	{
		mediapipe::StatusOrPoller faceLandmarks = m_Graph.AddOutputStreamPoller(m_FaceLandmarks_OutputStreamName);
		assert(faceLandmarks.ok());
		m_pFaceLandmarksPoller = std::make_unique<mediapipe::OutputStreamPoller>(std::move(faceLandmarks.value()));
	}

	// 设置图的输入流参数
	// 设置图的输入流队列中最大可排队Packet数
	//m_Graph.SetInputStreamMaxQueueSize(m_Video_InputStreamName,1);
	// 设置图的输入流队列添加新包的参数
	//m_Graph.SetGraphInputStreamAddMode(mediapipe::CalculatorGraph::GraphInputStreamAddMode::WAIT_TILL_NOT_FULL);

	MP_RETURN_IF_ERROR(m_Graph.StartRun({}));
	std::cout << "----------------Graph StartRun Success---------------------" << std::endl;
	return absl::OkStatus();
}

absl::Status GoogleMediapipeDetect::HolisticTrackingDetect::Mediapipe_RunMPPGraph_Direct(
	int image_width,
	int image_height,
	void* image_data,
	std::vector<cv::Point3f>& pose,
	std::vector<cv::Point2f>& leftHandPose,
	std::vector<cv::Point2f>& rightHandPose,
	std::vector<cv::Point3f>& facePose,
	bool show_result_image)
{
	m_Graph.WaitUntilIdle();

	/*----- 1 构造cv::Mat对象 -----*/
	cv::Mat camera_frame(cv::Size(image_width, image_height), CV_8UC3, (uchar*)image_data);
	
	// BGR转RGB
	cv::Mat camera_frame_RGB;
	cv::cvtColor(camera_frame, camera_frame_RGB, cv::COLOR_BGR2RGB);

	// 水平翻转
	//cv::flip(camera_frame_RGB, camera_frame_RGB, 1);

	//std::cout << "cv::Mat对象构建完成" << std::endl;

	/*----- 2 将OpenCV Mat转换为ImageFrame -----*/
	auto input_frame = absl::make_unique<mediapipe::ImageFrame>(
		mediapipe::ImageFormat::SRGB, camera_frame_RGB.cols, camera_frame_RGB.rows,
		mediapipe::ImageFrame::kDefaultAlignmentBoundary);
	//camera_frame_RGB.copyTo(mediapipe::formats::MatView(input_frame.get()));
	cv::Mat input_frame_mat = mediapipe::formats::MatView(input_frame.get());
	camera_frame_RGB.copyTo(input_frame_mat);

	//std::cout << "将OpenCV Mat转换为ImageFrame完成" << std::endl;

	/*----- 3 发送图片到图中推理 -----*/
	// 获取当前距离1970的微秒时间

	
	//size_t frame_timestamp_us = (double)cv::getTickCount() / (double)cv::getTickFrequency() * 1e6;

	 std::chrono::time_point<std::chrono::system_clock, std::chrono::microseconds> tpMicro
         = std::chrono::time_point_cast<std::chrono::microseconds>(std::chrono::system_clock::now());
	 size_t frame_timestamp_us = tpMicro.time_since_epoch().count();

	MP_RETURN_IF_ERROR(m_Graph.AddPacketToInputStream(
		m_Video_InputStreamName, mediapipe::Adopt(input_frame.release())
		.At(mediapipe::Timestamp(frame_timestamp_us))));

	

	/*----- 4 得到结果 -----*/
	// 1 视频输出结果帧
	
	// 等待图处于空闲状态
	

	if (show_result_image)
	{
		mediapipe::Packet videoPacket;
		if (!m_pVideoPoller->Next(&videoPacket))
		{
			return absl::InvalidArgumentError("no next packet");
		}

		if (show_result_image)
		{
			// 从视频输出获取mediapipe::ImageFrame结果
			auto& output_frame = videoPacket.Get<mediapipe::ImageFrame>();

			// 转换mediapipe::ImageFrame为cv::Mat
			cv::Mat output_frame_mat = mediapipe::formats::MatView(&output_frame);

			// 显示cv::Mat结果
			cv::cvtColor(output_frame_mat, output_frame_mat, cv::COLOR_RGB2BGR);
			cv::imshow("MediapipeHolistic", output_frame_mat);
		}
	}


	// 2 PoseLandmarks
	
	if(m_pPoseLandmarksPoller != nullptr)
	{
		mediapipe::Packet poseeLandmarksPacket;
		/*std::cout << "pose QueueSize" << m_pPoseLandmarksPoller->QueueSize() << std::endl;*/
		if (m_pPoseLandmarksPoller->QueueSize() != 0)
		{
			if (m_pPoseLandmarksPoller->Next(&poseeLandmarksPacket))
			{
				auto& output_landmarks = poseeLandmarksPacket.Get<mediapipe::NormalizedLandmarkList>();
				//std::cout << "PoseLandmarks size:" << output_landmarks.landmark_size() << std::endl;

				pose.clear();

				for (int i = 0; i < output_landmarks.landmark_size(); ++i)
				{
					cv::Point3f tempPoint3D;
					const mediapipe::NormalizedLandmark landmark = output_landmarks.landmark(i);
					tempPoint3D.x = landmark.x() * camera_frame.cols;
					tempPoint3D.y = landmark.y() * camera_frame.rows;
					tempPoint3D.z = landmark.z();
					pose.emplace_back(tempPoint3D);
				}
			}
		}
	}
	

	// 3 LeftHandLandmarks
	
	if(m_pLeftHandLandmarksPoller != nullptr)
	{
		mediapipe::Packet leftHandLandmarksPacket;	
		//std::cout << "left QueueSize" << m_pLeftHandLandmarksPoller->QueueSize() << std::endl;
		if (m_pLeftHandLandmarksPoller->QueueSize() > 0)
		{
			if (m_pLeftHandLandmarksPoller->Next(&leftHandLandmarksPacket))
			{
				auto& output_landmarks = leftHandLandmarksPacket.Get<mediapipe::NormalizedLandmarkList>();
				//std::cout << "LeftHandLandmarks size:" << output_landmarks.landmark_size() << std::endl;		
				leftHandPose.clear();
				for (int i = 0; i < output_landmarks.landmark_size(); ++i)
				{
					cv::Point2f tempPoint2D;
					const mediapipe::NormalizedLandmark landmark = output_landmarks.landmark(i);
					tempPoint2D.x = landmark.x() * camera_frame.cols;
					tempPoint2D.y = landmark.y() * camera_frame.rows;

					leftHandPose.emplace_back(tempPoint2D);
				}
				
			}
		}
	}
	


	// 4 RightHandLandmarks

	if(m_pRightHandLandmarksPoller != nullptr)
	{
		mediapipe::Packet rightHandLandmarksPacket;
		//std::cout << "right QueueSize" << m_pRightHandLandmarksPoller->QueueSize() << std::endl;
		if (m_pRightHandLandmarksPoller->QueueSize() > 0)
		{
			if (m_pRightHandLandmarksPoller->Next(&rightHandLandmarksPacket))
			{
				auto& output_landmarks = rightHandLandmarksPacket.Get<mediapipe::NormalizedLandmarkList>();
				//std::cout << "RightHandLandmarks size:" << output_landmarks.landmark_size() << std::endl;

				rightHandPose.clear();

				for (int i = 0; i < output_landmarks.landmark_size(); ++i)
				{
					cv::Point2f tempPoint2D;
					const mediapipe::NormalizedLandmark landmark = output_landmarks.landmark(i);
					tempPoint2D.x = landmark.x() * camera_frame.cols;
					tempPoint2D.y = landmark.y() * camera_frame.rows;
					
					rightHandPose.emplace_back(tempPoint2D);
				}
				
			}
		}
	}
	

	// 5 FaceLandmarks

	if(m_pFaceLandmarksPoller != nullptr)
	{
		mediapipe::Packet faceLandmarksPacket;
		//std::cout << "face QueueSize" << m_pFaceLandmarksPoller->QueueSize() << std::endl;
		if (m_pFaceLandmarksPoller->QueueSize() > 0)
		{
			if (m_pFaceLandmarksPoller->Next(&faceLandmarksPacket))
			{
				auto& output_landmarks = faceLandmarksPacket.Get<mediapipe::NormalizedLandmarkList>();
				//std::cout << "FaceLandmarks size:" << output_landmarks.landmark_size() << std::endl;
				facePose.clear();
				for (int i = 0; i < output_landmarks.landmark_size(); ++i)
				{
					cv::Point3f p;
					const mediapipe::NormalizedLandmark landmark = output_landmarks.landmark(i);
					p.x = landmark.x();
					p.y = landmark.y();
					p.z = landmark.z();
					facePose.emplace_back(p);
				}
			}
		}
	}

	return absl::OkStatus();
}




absl::Status GoogleMediapipeDetect::HolisticTrackingDetect::Mediapipe_ReleaseGraph()
{
	MP_RETURN_IF_ERROR(m_Graph.CloseInputStream(m_Video_InputStreamName));

	return m_Graph.WaitUntilDone();
}
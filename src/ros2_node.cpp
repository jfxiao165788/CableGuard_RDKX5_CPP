#include "cable_detector.hpp"

#include <cv_bridge/cv_bridge.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/string.hpp>

#include <memory>
#include <sstream>
#include <stdexcept>

class CableDetectionNode final : public rclcpp::Node {
 public:
  CableDetectionNode() : Node("cable_detection_cpp") {
    this->declare_parameter<std::string>("model_path", "");
    this->declare_parameter<std::string>("input_topic", "/camera/image_raw");
    this->declare_parameter<std::string>("result_topic", "/detection/results");
    this->declare_parameter<std::string>("annotated_topic", "/detection/annotated_image");
    this->declare_parameter<std::string>("mask_topic", "/detection/cable_mask");
    this->declare_parameter<std::string>("class_names", "burn,pr");
    this->declare_parameter<int>("inference_interval", 3);
    this->declare_parameter<double>("conf_thresh", 0.50);
    this->declare_parameter<double>("nms_thresh", 0.45);
    this->declare_parameter<bool>("use_cable_mask", true);
    this->declare_parameter<bool>("pre_mask_for_yolo", true);
    this->declare_parameter<bool>("allow_synthetic_backend", false);
    this->declare_parameter<std::string>("cpu_affinity", "");
    this->declare_parameter<int>("opencv_threads", 1);

    cable::DetectorConfig config;
    config.inference_interval = this->get_parameter("inference_interval").as_int();
    config.confidence_threshold = static_cast<float>(this->get_parameter("conf_thresh").as_double());
    config.nms_threshold = static_cast<float>(this->get_parameter("nms_thresh").as_double());
    config.use_cable_mask = this->get_parameter("use_cable_mask").as_bool();
    config.pre_mask_for_yolo = this->get_parameter("pre_mask_for_yolo").as_bool();
    const std::string classes = this->get_parameter("class_names").as_string();
    std::stringstream class_stream(classes);
    std::string class_name;
    config.class_names.clear();
    while (std::getline(class_stream, class_name, ',')) {
      if (!class_name.empty()) config.class_names.push_back(class_name);
    }
    if (config.class_names.empty()) config.class_names = {"burn", "pr"};

    const std::string model_path = this->get_parameter("model_path").as_string();
    const bool allow_synthetic = this->get_parameter("allow_synthetic_backend").as_bool();
    std::shared_ptr<cable::BpuBackend> backend = std::make_shared<cable::SyntheticBackend>();
    if (!model_path.empty()) {
      auto horizon = std::make_shared<cable::HorizonBpuBackend>(model_path);
      if (horizon->is_ready()) backend = horizon;
      else if (allow_synthetic) {
        RCLCPP_WARN(this->get_logger(), "Horizon backend unavailable; using explicitly enabled synthetic CPU backend");
      } else {
        throw std::runtime_error(
            "Horizon backend unavailable for model_path; install the X5 SDK adapter "
            "or set allow_synthetic_backend:=true for development only");
      }
    }
    detector_ = std::make_unique<cable::CableDetector>(config, backend);

    cable::configure_opencv_threads(this->get_parameter("opencv_threads").as_int());
    const std::string affinity = this->get_parameter("cpu_affinity").as_string();
    if (!affinity.empty()) cable::set_process_affinity(affinity);

    const auto input_topic = this->get_parameter("input_topic").as_string();
    image_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
        this->get_parameter("annotated_topic").as_string(), rclcpp::SensorDataQoS());
    result_pub_ = this->create_publisher<std_msgs::msg::String>(
        this->get_parameter("result_topic").as_string(), 10);
    center_pub_ = this->create_publisher<std_msgs::msg::Bool>(
        "/detection/defect_in_center", 10);
    detected_pub_ = this->create_publisher<std_msgs::msg::Bool>(
        "/detection/defect_detected", 10);
    mask_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
        this->get_parameter("mask_topic").as_string(), rclcpp::SensorDataQoS());
    image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
        input_topic, rclcpp::SensorDataQoS(),
        std::bind(&CableDetectionNode::on_image, this, std::placeholders::_1));
    RCLCPP_INFO(this->get_logger(), "C++ detector ready: backend=%s interval=%d",
                detector_->backend()->name().c_str(), config.inference_interval);
  }

 private:
  void on_image(const sensor_msgs::msg::Image::SharedPtr message) {
    try {
      const cv::Mat frame = cv_bridge::toCvShare(message, "bgr8")->image;
      const auto result = detector_->process(frame);
      std_msgs::msg::String result_message;
      result_message.data = detector_->detections_to_json(result.detections);
      result_pub_->publish(result_message);
      std_msgs::msg::Bool detected_message;
      detected_message.data = result.defect_detected;
      detected_pub_->publish(detected_message);
      std_msgs::msg::Bool center_message;
      center_message.data = result.defect_in_center;
      center_pub_->publish(center_message);
      auto annotated = cv_bridge::CvImage(message->header, "bgr8", result.annotated).toImageMsg();
      image_pub_->publish(*annotated);
      if (!result.cable_mask.empty()) {
        auto mask = cv_bridge::CvImage(message->header, "mono8", result.cable_mask).toImageMsg();
        mask_pub_->publish(*mask);
      }
    } catch (const std::exception& error) {
      RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 5000, "Detection failed: %s", error.what());
    }
  }

  std::unique_ptr<cable::CableDetector> detector_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr mask_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr result_pub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr center_pub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr detected_pub_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<CableDetectionNode>());
  rclcpp::shutdown();
  return 0;
}

#include "cable_detector.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace cable {

std::vector<RawDetection> SyntheticBackend::infer(
    const cv::Mat& image, const std::vector<std::string>& class_names,
    float confidence_threshold, float /*nms_threshold*/) {
  if (image.empty() || class_names.empty()) return {};

  cv::Mat bgr_gray;
  cv::cvtColor(image, bgr_gray, cv::COLOR_BGR2GRAY);
  cv::Mat bright;
  cv::threshold(bgr_gray, bright, 185, 255, cv::THRESH_BINARY);
  cv::morphologyEx(bright, bright, cv::MORPH_OPEN,
                   cv::getStructuringElement(cv::MORPH_ELLIPSE, {3, 3}));
  cv::morphologyEx(bright, bright, cv::MORPH_CLOSE,
                   cv::getStructuringElement(cv::MORPH_ELLIPSE, {5, 5}));

  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(bright, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
  std::vector<RawDetection> output;
  const double image_area = static_cast<double>(image.rows) * image.cols;
  for (const auto& contour : contours) {
    const double area = cv::contourArea(contour);
    if (area < std::max(12.0, image_area * 0.00002)) continue;
    const cv::Rect rect = cv::boundingRect(contour);
    if (rect.width < 3 || rect.height < 3) continue;
    const float confidence = std::min(0.99F, 0.70F +
        static_cast<float>(std::min(0.29, area / std::max(1.0, image_area * 0.01))));
    if (confidence < confidence_threshold) continue;
    output.push_back({static_cast<float>(rect.x), static_cast<float>(rect.y),
                      static_cast<float>(rect.x + rect.width),
                      static_cast<float>(rect.y + rect.height), confidence, 0});
  }
  return output;
}

struct HorizonBpuBackend::Impl {
  std::string model_path;
  bool ready{false};
};

HorizonBpuBackend::HorizonBpuBackend(const std::string& model_path)
    : impl_(std::make_unique<Impl>()) {
  impl_->model_path = model_path;
#if defined(CABLE_ENABLE_HORIZON_BPU) && __has_include(<hb_dnn.h>)
  // The board-specific implementation is intentionally isolated here. The
  // public Horizon SDK has changed names across X5 images; wire hb_dnn.h and
  // the matching runtime in this file when building on the target image.
  // The detector and ROS2 node do not depend on this compile-time branch.
  impl_->ready = false;
#else
  impl_->ready = false;
#endif
}

HorizonBpuBackend::~HorizonBpuBackend() = default;

bool HorizonBpuBackend::is_ready() const { return impl_ && impl_->ready; }

std::string HorizonBpuBackend::name() const { return "horizon-bpu"; }

std::vector<RawDetection> HorizonBpuBackend::infer(
    const cv::Mat& /*image*/, const std::vector<std::string>& /*class_names*/,
    float /*confidence_threshold*/, float /*nms_threshold*/) {
  throw std::runtime_error(
      "HorizonBpuBackend is not enabled in this build. Rebuild on RDK X5 "
      "with the matching hb_dnn SDK adapter in src/bpu_backend.cpp.");
}

}  // namespace cable

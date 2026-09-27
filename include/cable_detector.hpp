#pragma once

#include <opencv2/core.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace cable {

struct BBox {
  int x1{0};
  int y1{0};
  int x2{0};
  int y2{0};
};

struct Detection {
  std::string class_name;
  float confidence{0.0F};
  BBox bbox;
  cv::Point2f center;
  bool refined{false};
  float mask_ratio{0.0F};
  float refine_area{0.0F};
  std::vector<cv::Point> rotated_box;
  std::vector<cv::Point> contour;
};

struct RawDetection {
  float x1{0.0F};
  float y1{0.0F};
  float x2{0.0F};
  float y2{0.0F};
  float confidence{0.0F};
  int class_id{0};
};

struct DetectorConfig {
  int input_size{640};
  float confidence_threshold{0.50F};
  float nms_threshold{0.45F};
  int inference_interval{3};
  int background_gray{127};
  int cable_dark_l_threshold{90};
  int cable_dark_v_threshold{90};
  int cable_extra_pad{10};
  float bbox_mask_min_ratio{0.12F};
  float min_cable_span_ratio{0.35F};
  int roi_pad{24};
  int min_defect_area{12};
  float refine_blackhat_weight{0.50F};
  float refine_gradient_weight{0.30F};
  float refine_contrast_weight{0.20F};
  int center_threshold{80};
  int center_confirm_frames{3};
  bool use_cable_mask{true};
  bool pre_mask_for_yolo{true};
  bool show_cable_mask{true};
  std::vector<std::string> class_names{"burn", "pr"};
};

struct FrameResult {
  std::vector<Detection> detections;
  cv::Mat cable_mask;
  cv::Mat annotated;
  bool defect_detected{false};
  bool defect_in_center{false};
  int frame_index{0};
  bool inference_ran{false};
};

class BpuBackend {
 public:
  virtual ~BpuBackend() = default;
  virtual bool is_ready() const = 0;
  virtual std::string name() const = 0;
  virtual std::vector<RawDetection> infer(const cv::Mat& bgr_letterboxed,
                                          const std::vector<std::string>& class_names,
                                          float confidence_threshold,
                                          float nms_threshold) = 0;
};

// Deterministic local backend used by the benchmark and by development machines
// without Horizon runtime. It finds bright/contrasting regions on the cable band.
class SyntheticBackend final : public BpuBackend {
 public:
  bool is_ready() const override { return true; }
  std::string name() const override { return "synthetic-cpu"; }
  std::vector<RawDetection> infer(const cv::Mat& bgr_letterboxed,
                                  const std::vector<std::string>& class_names,
                                  float confidence_threshold,
                                  float nms_threshold) override;
};

// Adapter boundary for the RDK X5. The implementation is compiled without
// Horizon headers by default, so this project remains buildable on a PC.
// Define CABLE_ENABLE_HORIZON_BPU and provide the board SDK to enable it.
class HorizonBpuBackend final : public BpuBackend {
 public:
  explicit HorizonBpuBackend(const std::string& model_path);
  ~HorizonBpuBackend() override;
  bool is_ready() const override;
  std::string name() const override;
  std::vector<RawDetection> infer(const cv::Mat& bgr_letterboxed,
                                  const std::vector<std::string>& class_names,
                                  float confidence_threshold,
                                  float nms_threshold) override;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

class CableDetector {
 public:
  explicit CableDetector(DetectorConfig config = {},
                         std::shared_ptr<BpuBackend> backend = nullptr);

  FrameResult process(const cv::Mat& bgr_frame);
  cv::Mat extract_cable_band_mask(const cv::Mat& frame) const;
  cv::Mat preprocess_bgr(const cv::Mat& frame, float& scale, int& x_shift,
                         int& y_shift) const;
  std::vector<Detection> postprocess(const std::vector<RawDetection>& raw,
                                    float scale, int x_shift, int y_shift,
                                    cv::Size original_size) const;
  std::string detections_to_json(const std::vector<Detection>& detections) const;
  cv::Mat draw_annotations(const cv::Mat& frame,
                          const std::vector<Detection>& detections,
                          const cv::Mat& cable_mask) const;

  const DetectorConfig& config() const { return config_; }
  const std::shared_ptr<BpuBackend>& backend() const { return backend_; }
  uint64_t processed_frames() const { return frame_count_; }
  uint64_t reused_frames() const { return reused_frames_; }

 private:
  static int odd_kernel(float value, int min_value, int max_value);
  cv::Mat fit_cable_band_from_dark_mask(const cv::Mat& dark_mask) const;
  cv::Mat extract_cable_mask_fallback(const cv::Mat& frame,
                                      const cv::Mat& dark_mask) const;
  cv::Mat extract_roi_cable_mask(const cv::Mat& gray) const;
  static cv::Mat normalize_score(const cv::Mat& input);
  static float bbox_mask_ratio(const BBox& bbox, const cv::Mat& mask);
  void filter_detections_by_cable_mask(std::vector<Detection>& detections,
                                       const cv::Mat& cable_mask) const;
  void refine_detection(const cv::Mat& frame, Detection& detection,
                        const cv::Mat& cable_mask) const;
  bool is_defect_in_center(const std::vector<Detection>& detections) const;

  DetectorConfig config_;
  std::shared_ptr<BpuBackend> backend_;
  std::vector<Detection> last_detections_;
  cv::Mat last_cable_mask_;
  cv::Size last_frame_size_;
  int last_inference_frame_{-1};
  int center_hit_count_{0};
  uint64_t frame_count_{0};
  uint64_t reused_frames_{0};
};

void set_process_affinity(const std::string& cpu_list);
void configure_opencv_threads(int threads);

}  // namespace cable

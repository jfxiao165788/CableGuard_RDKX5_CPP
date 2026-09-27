#include "cable_detector.hpp"
#include "json_writer.hpp"

#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <numeric>
#include <sched.h>
#include <sstream>
#include <stdexcept>
#include <thread>

namespace cable {
namespace {

float percentile(std::vector<float> values, float p) {
  if (values.empty()) return 0.0F;
  std::sort(values.begin(), values.end());
  const float index = std::clamp(p, 0.0F, 100.0F) *
                      static_cast<float>(values.size() - 1) / 100.0F;
  const auto lo = static_cast<size_t>(std::floor(index));
  const auto hi = static_cast<size_t>(std::ceil(index));
  if (lo == hi) return values[lo];
  const float fraction = index - static_cast<float>(lo);
  return values[lo] * (1.0F - fraction) + values[hi] * fraction;
}

int clamp_int(int value, int low, int high) {
  return std::max(low, std::min(value, high));
}

}  // namespace

CableDetector::CableDetector(DetectorConfig config,
                             std::shared_ptr<BpuBackend> backend)
    : config_(std::move(config)), backend_(std::move(backend)) {
  if (config_.class_names.empty()) config_.class_names = {"burn", "pr"};
  config_.inference_interval = std::max(1, config_.inference_interval);
  if (!backend_) backend_ = std::make_shared<SyntheticBackend>();
}

int CableDetector::odd_kernel(float value, int min_value, int max_value) {
  int result = clamp_int(static_cast<int>(value), min_value, max_value);
  if ((result & 1) == 0) ++result;
  return std::min(result, max_value | 1);
}

cv::Mat CableDetector::fit_cable_band_from_dark_mask(
    const cv::Mat& dark_mask) const {
  const int h = dark_mask.rows;
  const int w = dark_mask.cols;
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(dark_mask, contours, cv::RETR_EXTERNAL,
                   cv::CHAIN_APPROX_SIMPLE);
  if (contours.empty()) return {};

  int best_index = -1;
  double best_score = -1.0;
  for (int i = 0; i < static_cast<int>(contours.size()); ++i) {
    const double area = cv::contourArea(contours[i]);
    if (area < 0.002 * h * w) continue;
    const cv::RotatedRect rect = cv::minAreaRect(contours[i]);
    const double rw = rect.size.width;
    const double rh = rect.size.height;
    if (rw < 2.0 || rh < 2.0) continue;
    const double aspect = std::max(rw, rh) / std::max(1.0, std::min(rw, rh));
    const cv::Rect bounds = cv::boundingRect(contours[i]);
    const double span = std::max(static_cast<double>(bounds.width) / w,
                                 static_cast<double>(bounds.height) / h);
    const double score = area * (1.0 + std::min(aspect, 14.0)) * (0.6 + span);
    if (score > best_score) {
      best_score = score;
      best_index = i;
    }
  }
  if (best_index < 0 || contours[best_index].size() < 8) return {};

  cv::Vec4f line;
  cv::fitLine(contours[best_index], line, cv::DIST_L2, 0, 0.01, 0.01);
  cv::Point2f direction(line[0], line[1]);
  const float norm = std::sqrt(direction.dot(direction));
  if (norm < 1e-6F) return {};
  direction *= 1.0F / norm;
  const cv::Point2f normal(-direction.y, direction.x);
  const cv::Point2f origin(line[2], line[3]);

  std::vector<float> projections;
  std::vector<float> distances;
  projections.reserve(contours[best_index].size());
  distances.reserve(contours[best_index].size());
  for (const cv::Point& point : contours[best_index]) {
    const cv::Point2f relative = cv::Point2f(point) - origin;
    projections.push_back(relative.dot(direction));
    distances.push_back(std::abs(relative.dot(normal)));
  }
  const float t_min = percentile(projections, 2.0F);
  const float t_max = percentile(projections, 98.0F);
  if (t_max - t_min < std::min(w, h) * 0.25F) return {};

  const int min_half = std::max(8, static_cast<int>(h * 0.05F));
  const int max_half = std::max(12, static_cast<int>(h * 0.12F));
  const int half_width = clamp_int(
      static_cast<int>(percentile(distances, 82.0F) + config_.cable_extra_pad),
      min_half, max_half);
  const float extend = std::max(12.0F, std::min(w, h) * 0.06F);
  const cv::Point2f p1 = origin + direction * (t_min - extend);
  const cv::Point2f p2 = origin + direction * (t_max + extend);
  std::vector<cv::Point> polygon{
      p1 + normal * static_cast<float>(half_width),
      p2 + normal * static_cast<float>(half_width),
      p2 - normal * static_cast<float>(half_width),
      p1 - normal * static_cast<float>(half_width)};
  for (auto& point : polygon) {
    point.x = clamp_int(point.x, 0, w - 1);
    point.y = clamp_int(point.y, 0, h - 1);
  }
  cv::Mat mask = cv::Mat::zeros(h, w, CV_8UC1);
  cv::fillConvexPoly(mask, polygon, 255);
  const int close_k = odd_kernel(std::min(h, w) * 0.04F, 3, 13);
  cv::morphologyEx(mask, mask, cv::MORPH_CLOSE,
                   cv::getStructuringElement(cv::MORPH_ELLIPSE,
                                             {close_k, close_k}));
  return mask;
}

cv::Mat CableDetector::extract_cable_mask_fallback(
    const cv::Mat& frame, const cv::Mat& dark_mask) const {
  const int h = frame.rows;
  const int w = frame.cols;
  cv::Mat dark;
  const int close_k = odd_kernel(std::min(h, w) * 0.12F, 7, 31);
  cv::morphologyEx(dark_mask, dark, cv::MORPH_CLOSE,
                   cv::getStructuringElement(cv::MORPH_ELLIPSE,
                                             {close_k, close_k}), cv::Point(-1, -1), 2);
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(dark, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
  if (contours.empty()) return cv::Mat(h, w, CV_8UC1, cv::Scalar(255)).clone();

  int best_index = 0;
  double best_score = -1.0;
  for (int i = 0; i < static_cast<int>(contours.size()); ++i) {
    const double area = cv::contourArea(contours[i]);
    if (area < 0.002 * h * w) continue;
    const cv::RotatedRect rect = cv::minAreaRect(contours[i]);
    const double aspect = std::max(static_cast<double>(rect.size.width),
                                   static_cast<double>(rect.size.height)) /
                         std::max(1.0, static_cast<double>(std::min(rect.size.width, rect.size.height)));
    const cv::Rect bounds = cv::boundingRect(contours[i]);
    const double span = std::max(static_cast<double>(bounds.width) / w,
                                 static_cast<double>(bounds.height) / h);
    const double score = area * (1.0 + 0.7 * std::min(aspect, 12.0) + span);
    if (score > best_score) {
      best_score = score;
      best_index = i;
    }
  }
  cv::Mat mask = cv::Mat::zeros(h, w, CV_8UC1);
  cv::drawContours(mask, contours, best_index, 255, cv::FILLED);
  const cv::Mat fitted = fit_cable_band_from_dark_mask(dark);
  if (!fitted.empty()) return fitted;
  const int dilate_k = odd_kernel(std::min(h, w) * 0.08F, 5, 19);
  const auto kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE,
                                                 {dilate_k, dilate_k});
  cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel, cv::Point(-1, -1), 2);
  cv::dilate(mask, mask, kernel);
  return mask;
}

cv::Mat CableDetector::extract_cable_band_mask(const cv::Mat& frame) const {
  if (frame.empty()) return {};
  cv::Mat hsv, lab;
  cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
  cv::cvtColor(frame, lab, cv::COLOR_BGR2Lab);
  cv::Mat l_channel, v_channel, dark_l, dark_v, dark;
  cv::extractChannel(lab, l_channel, 0);
  cv::extractChannel(hsv, v_channel, 2);
  cv::compare(l_channel, config_.cable_dark_l_threshold, dark_l, cv::CMP_LT);
  cv::compare(v_channel, config_.cable_dark_v_threshold, dark_v, cv::CMP_LT);
  cv::bitwise_or(dark_l, dark_v, dark);
  cv::medianBlur(dark, dark, 5);

  const int close_w = odd_kernel(frame.cols * 0.18F, 9, 61);
  const int close_h = odd_kernel(frame.rows * 0.06F, 3, 17);
  cv::morphologyEx(dark, dark, cv::MORPH_CLOSE,
                   cv::getStructuringElement(cv::MORPH_RECT,
                                             {close_w, close_h}), cv::Point(-1, -1), 2);
  cv::morphologyEx(dark, dark, cv::MORPH_OPEN,
                   cv::getStructuringElement(cv::MORPH_ELLIPSE, {5, 5}));

  cv::Mat fitted = fit_cable_band_from_dark_mask(dark);
  if (!fitted.empty()) {
    const double ratio = static_cast<double>(cv::countNonZero(fitted)) /
                         static_cast<double>(fitted.total());
    if (ratio >= 0.02 && ratio <= 0.70) return fitted;
  }
  return extract_cable_mask_fallback(frame, dark);
}

cv::Mat CableDetector::preprocess_bgr(const cv::Mat& frame, float& scale,
                                      int& x_shift, int& y_shift) const {
  if (frame.empty()) return {};
  scale = std::min(static_cast<float>(config_.input_size) / frame.rows,
                   static_cast<float>(config_.input_size) / frame.cols);
  const int nw = std::max(1, static_cast<int>(frame.cols * scale));
  const int nh = std::max(1, static_cast<int>(frame.rows * scale));
  x_shift = (config_.input_size - nw) / 2;
  y_shift = (config_.input_size - nh) / 2;
  cv::Mat resized;
  cv::resize(frame, resized, {nw, nh}, 0.0, 0.0, cv::INTER_LINEAR);
  cv::Mat padded;
  cv::copyMakeBorder(resized, padded, y_shift,
                     config_.input_size - nh - y_shift, x_shift,
                     config_.input_size - nw - x_shift, cv::BORDER_CONSTANT,
                     cv::Scalar(config_.background_gray, config_.background_gray,
                                config_.background_gray));
  return padded;
}

std::vector<Detection> CableDetector::postprocess(
    const std::vector<RawDetection>& raw, float scale, int x_shift, int y_shift,
    cv::Size original_size) const {
  std::vector<cv::Rect> boxes;
  std::vector<float> scores;
  std::vector<int> ids;
  boxes.reserve(raw.size());
  scores.reserve(raw.size());
  ids.reserve(raw.size());
  for (const auto& d : raw) {
    if (d.class_id < 0 || d.class_id >= static_cast<int>(config_.class_names.size()) ||
        d.confidence < config_.confidence_threshold) continue;
    const int x = static_cast<int>(std::round(d.x1));
    const int y = static_cast<int>(std::round(d.y1));
    const int width = static_cast<int>(std::round(d.x2 - d.x1));
    const int height = static_cast<int>(std::round(d.y2 - d.y1));
    if (width <= 0 || height <= 0) continue;
    boxes.emplace_back(x, y, width, height);
    scores.push_back(d.confidence);
    ids.push_back(d.class_id);
  }
  std::vector<int> selected;
  if (!boxes.empty()) cv::dnn::NMSBoxes(boxes, scores, config_.confidence_threshold,
                                        config_.nms_threshold, selected);
  std::vector<Detection> output;
  output.reserve(selected.size());
  for (const int index : selected) {
    const auto& box = boxes[index];
    const float x1f = (box.x - x_shift) / scale;
    const float y1f = (box.y - y_shift) / scale;
    const float x2f = (box.x + box.width - x_shift) / scale;
    const float y2f = (box.y + box.height - y_shift) / scale;
    BBox bbox{
        clamp_int(static_cast<int>(std::floor(x1f)), 0, original_size.width),
        clamp_int(static_cast<int>(std::floor(y1f)), 0, original_size.height),
        clamp_int(static_cast<int>(std::ceil(x2f)), 0, original_size.width),
        clamp_int(static_cast<int>(std::ceil(y2f)), 0, original_size.height)};
    if (bbox.x2 <= bbox.x1 || bbox.y2 <= bbox.y1) continue;
    Detection detection;
    detection.class_name = config_.class_names[ids[index]];
    detection.confidence = scores[index];
    detection.bbox = bbox;
    detection.center = {(bbox.x1 + bbox.x2) * 0.5F,
                        (bbox.y1 + bbox.y2) * 0.5F};
    output.push_back(std::move(detection));
  }
  return output;
}

float CableDetector::bbox_mask_ratio(const BBox& bbox, const cv::Mat& mask) {
  const int x1 = clamp_int(bbox.x1, 0, mask.cols - 1);
  const int y1 = clamp_int(bbox.y1, 0, mask.rows - 1);
  const int x2 = clamp_int(bbox.x2, 0, mask.cols);
  const int y2 = clamp_int(bbox.y2, 0, mask.rows);
  if (x2 <= x1 || y2 <= y1) return 0.0F;
  const cv::Mat roi = mask(cv::Range(y1, y2), cv::Range(x1, x2));
  if (roi.empty()) return 0.0F;
  return static_cast<float>(cv::countNonZero(roi)) /
         static_cast<float>(roi.total());
}

void CableDetector::filter_detections_by_cable_mask(
    std::vector<Detection>& detections, const cv::Mat& cable_mask) const {
  if (cable_mask.empty()) return;
  std::vector<Detection> filtered;
  filtered.reserve(detections.size());
  for (auto& detection : detections) {
    const int cx = clamp_int(static_cast<int>(detection.center.x), 0, cable_mask.cols - 1);
    const int cy = clamp_int(static_cast<int>(detection.center.y), 0, cable_mask.rows - 1);
    const bool center_on_cable = cable_mask.at<uint8_t>(cy, cx) > 0;
    const float ratio = bbox_mask_ratio(detection.bbox, cable_mask);
    if (center_on_cable || ratio >= config_.bbox_mask_min_ratio) {
      detection.mask_ratio = ratio;
      filtered.push_back(std::move(detection));
    }
  }
  detections = std::move(filtered);
}

cv::Mat CableDetector::extract_roi_cable_mask(const cv::Mat& gray) const {
  cv::Mat blur, mask;
  cv::GaussianBlur(gray, blur, {5, 5}, 0);
  cv::threshold(blur, mask, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);
  const int close_k = odd_kernel(std::min(gray.rows, gray.cols) * 0.12F, 5, 17);
  cv::morphologyEx(mask, mask, cv::MORPH_CLOSE,
                   cv::getStructuringElement(cv::MORPH_ELLIPSE,
                                             {close_k, close_k}), cv::Point(-1, -1), 2);
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
  if (contours.empty()) return {};
  int index = 0;
  double area = 0.0;
  for (int i = 0; i < static_cast<int>(contours.size()); ++i) {
    const double candidate = cv::contourArea(contours[i]);
    if (candidate > area) { area = candidate; index = i; }
  }
  if (area < std::max(20.0, 0.01 * gray.rows * gray.cols)) return {};
  cv::Mat cable_mask = cv::Mat::zeros(mask.size(), CV_8UC1);
  cv::drawContours(cable_mask, contours, index, 255, cv::FILLED);
  return cable_mask;
}

cv::Mat CableDetector::normalize_score(const cv::Mat& input) {
  if (input.empty()) return {};
  double min_value = 0.0, max_value = 0.0;
  cv::minMaxLoc(input, &min_value, &max_value);
  if (max_value - min_value < 1e-6) return cv::Mat::zeros(input.size(), CV_32FC1);
  cv::Mat output;
  cv::normalize(input, output, 0.0, 1.0, cv::NORM_MINMAX, CV_32F);
  return output;
}

void CableDetector::refine_detection(const cv::Mat& frame, Detection& detection,
                                     const cv::Mat& cable_mask) const {
  const int rx1 = std::max(0, detection.bbox.x1 - config_.roi_pad);
  const int ry1 = std::max(0, detection.bbox.y1 - config_.roi_pad);
  const int rx2 = std::min(frame.cols, detection.bbox.x2 + config_.roi_pad);
  const int ry2 = std::min(frame.rows, detection.bbox.y2 + config_.roi_pad);
  if (rx2 <= rx1 || ry2 <= ry1) return;
  const cv::Mat roi = frame(cv::Range(ry1, ry2), cv::Range(rx1, rx2));
  cv::Mat gray, blur;
  cv::cvtColor(roi, gray, cv::COLOR_BGR2GRAY);
  cv::GaussianBlur(gray, blur, {5, 5}, 0);
  cv::Mat roi_mask = extract_roi_cable_mask(gray);
  if (roi_mask.empty() && !cable_mask.empty())
    roi_mask = cable_mask(cv::Range(ry1, ry2), cv::Range(rx1, rx2)).clone();
  if (roi_mask.empty() || cv::countNonZero(roi_mask) == 0) return;

  const int blackhat_k = odd_kernel(std::min(gray.rows, gray.cols) * 0.16F, 7, 21);
  cv::Mat blackhat;
  cv::morphologyEx(blur, blackhat, cv::MORPH_BLACKHAT,
                   cv::getStructuringElement(cv::MORPH_ELLIPSE,
                                             {blackhat_k, blackhat_k}));
  cv::Mat grad_x, grad_y, grad_mag;
  cv::Sobel(blur, grad_x, CV_32F, 1, 0, 3);
  cv::Sobel(blur, grad_y, CV_32F, 0, 1, 3);
  cv::magnitude(grad_x, grad_y, grad_mag);
  cv::Mat local_blur, contrast;
  cv::GaussianBlur(blur, local_blur, {15, 15}, 0);
  cv::absdiff(blur, local_blur, contrast);

  cv::Mat score = config_.refine_blackhat_weight * normalize_score(blackhat) +
                  config_.refine_gradient_weight * normalize_score(grad_mag) +
                  config_.refine_contrast_weight * normalize_score(contrast);
  score = cv::min(score * 255.0F, 255.0F);
  score.convertTo(score, CV_8U);
  cv::bitwise_and(score, score, score, roi_mask);
  if (cv::countNonZero(score) < config_.min_defect_area) return;

  cv::Mat defect_mask;
  cv::threshold(score, defect_mask, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
  cv::bitwise_and(defect_mask, defect_mask, defect_mask, roi_mask);
  cv::morphologyEx(defect_mask, defect_mask, cv::MORPH_CLOSE,
                   cv::getStructuringElement(cv::MORPH_RECT, {5, 5}),
                   cv::Point(-1, -1), 2);
  cv::morphologyEx(defect_mask, defect_mask, cv::MORPH_OPEN,
                   cv::getStructuringElement(cv::MORPH_RECT, {3, 3}));
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(defect_mask, contours, cv::RETR_EXTERNAL,
                   cv::CHAIN_APPROX_SIMPLE);
  std::vector<std::vector<cv::Point>> valid;
  for (const auto& contour : contours) {
    if (cv::contourArea(contour) >= config_.min_defect_area) valid.push_back(contour);
  }
  if (valid.empty()) return;
  std::vector<cv::Point> points;
  for (const auto& contour : valid) points.insert(points.end(), contour.begin(), contour.end());
  const cv::RotatedRect rect = cv::minAreaRect(points);
  cv::Point2f box_points[4];
  rect.points(box_points);
  std::vector<cv::Point> global_box;
  global_box.reserve(4);
  for (const auto& point : box_points) global_box.emplace_back(
      clamp_int(static_cast<int>(std::lround(point.x)) + rx1, 0, frame.cols),
      clamp_int(static_cast<int>(std::lround(point.y)) + ry1, 0, frame.rows));
  int min_x = frame.cols, min_y = frame.rows, max_x = 0, max_y = 0;
  for (const auto& point : global_box) {
    min_x = std::min(min_x, point.x); min_y = std::min(min_y, point.y);
    max_x = std::max(max_x, point.x); max_y = std::max(max_y, point.y);
  }
  if (max_x <= min_x || max_y <= min_y) return;
  detection.bbox = {min_x, min_y, max_x, max_y};
  detection.center = {rect.center.x + rx1, rect.center.y + ry1};
  detection.rotated_box = std::move(global_box);
  std::vector<cv::Point> hull;
  cv::convexHull(points, hull);
  for (auto& point : hull) { point.x += rx1; point.y += ry1; }
  detection.contour = std::move(hull);
  detection.refined = true;
  detection.refine_area = 0.0F;
  for (const auto& contour : valid) detection.refine_area += static_cast<float>(cv::contourArea(contour));
}

bool CableDetector::is_defect_in_center(
    const std::vector<Detection>& detections) const {
  if (detections.empty() || last_frame_size_.width <= 0) return false;
  const auto best = std::max_element(detections.begin(), detections.end(),
      [](const Detection& a, const Detection& b) { return a.confidence < b.confidence; });
  return std::abs(best->center.x - last_frame_size_.width * 0.5F) <
         config_.center_threshold;
}

cv::Mat CableDetector::draw_annotations(
    const cv::Mat& frame, const std::vector<Detection>& detections,
    const cv::Mat& cable_mask) const {
  cv::Mat visualization = frame.clone();
  if (config_.show_cable_mask && !cable_mask.empty()) {
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(cable_mask, contours, cv::RETR_EXTERNAL,
                     cv::CHAIN_APPROX_SIMPLE);
    cv::drawContours(visualization, contours, -1, {0, 255, 255}, 1);
  }
  const std::vector<cv::Scalar> colors{{0, 255, 0}, {255, 0, 0},
                                       {0, 165, 255}, {255, 255, 0}};
  for (const auto& detection : detections) {
    const auto it = std::find(config_.class_names.begin(), config_.class_names.end(),
                              detection.class_name);
    const size_t color_index = it == config_.class_names.end() ? 0 :
        static_cast<size_t>(std::distance(config_.class_names.begin(), it));
    const auto color = colors[color_index % colors.size()];
    cv::rectangle(visualization, {detection.bbox.x1, detection.bbox.y1},
                  {detection.bbox.x2, detection.bbox.y2}, color, 2);
    if (detection.rotated_box.size() == 4)
      cv::polylines(visualization, detection.rotated_box, true, {0, 255, 255}, 2);
    if (!detection.contour.empty())
      cv::drawContours(visualization, std::vector<std::vector<cv::Point>>{detection.contour},
                       -1, {255, 255, 0}, 1);
    std::ostringstream label;
    label << detection.class_name << ":" << std::fixed << std::setprecision(2)
          << detection.confidence << (detection.refined ? " R" : "");
    cv::putText(visualization, label.str(),
                {detection.bbox.x1, std::max(detection.bbox.y1 - 6, 14)},
                cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv::LINE_AA);
    cv::drawMarker(visualization,
                   {static_cast<int>(detection.center.x), static_cast<int>(detection.center.y)},
                   color, cv::MARKER_CROSS, 12, 1);
  }
  if (last_frame_size_.width > 0 && last_frame_size_.height > 0)
    cv::drawMarker(visualization, {last_frame_size_.width / 2,
                                   last_frame_size_.height / 2},
                   {0, 255, 255}, cv::MARKER_CROSS, 30, 2);
  return visualization;
}

FrameResult CableDetector::process(const cv::Mat& bgr_frame) {
  if (bgr_frame.empty()) throw std::invalid_argument("empty camera frame");
  const cv::Size previous_size = last_frame_size_;
  last_frame_size_ = bgr_frame.size();
  const int frame_index = static_cast<int>(frame_count_);
  const bool shape_changed = previous_size.width > 0 && previous_size != bgr_frame.size();
  const bool should_infer = last_inference_frame_ < 0 || shape_changed ||
      (frame_index % config_.inference_interval == 0);
  cv::Mat cable_mask;
  std::vector<Detection> detections;
  if (should_infer) {
    if (config_.use_cable_mask) cable_mask = extract_cable_band_mask(bgr_frame);
    cv::Mat input = bgr_frame;
    if (config_.pre_mask_for_yolo && !cable_mask.empty()) {
      input = bgr_frame.clone();
      input.setTo(cv::Scalar(config_.background_gray, config_.background_gray,
                             config_.background_gray), cable_mask == 0);
    }
    float scale = 1.0F;
    int x_shift = 0, y_shift = 0;
    const cv::Mat letterboxed = preprocess_bgr(input, scale, x_shift, y_shift);
    const auto raw = backend_->infer(letterboxed, config_.class_names,
                                     config_.confidence_threshold,
                                     config_.nms_threshold);
    detections = postprocess(raw, scale, x_shift, y_shift, bgr_frame.size());
    if (config_.use_cable_mask && !cable_mask.empty())
      filter_detections_by_cable_mask(detections, cable_mask);
    for (auto& detection : detections) refine_detection(bgr_frame, detection, cable_mask);
    last_detections_ = detections;
    last_cable_mask_ = cable_mask.clone();
    last_inference_frame_ = frame_index;
  } else {
    detections = last_detections_;
    cable_mask = last_cable_mask_;
    ++reused_frames_;
  }
  ++frame_count_;
  const bool now_in_center = is_defect_in_center(detections);
  center_hit_count_ = now_in_center ? center_hit_count_ + 1 : 0;
  FrameResult result;
  result.detections = std::move(detections);
  result.cable_mask = cable_mask;
  result.defect_detected = !result.detections.empty();
  result.defect_in_center = center_hit_count_ >= config_.center_confirm_frames;
  result.frame_index = frame_index;
  result.inference_ran = should_infer;
  result.annotated = draw_annotations(bgr_frame, result.detections, result.cable_mask);
  return result;
}

std::string CableDetector::detections_to_json(
    const std::vector<Detection>& detections) const {
  return detections_json(detections);
}

void configure_opencv_threads(int threads) {
  cv::setNumThreads(std::max(1, threads));
}

void set_process_affinity(const std::string& cpu_list) {
#if defined(__linux__)
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  std::stringstream stream(cpu_list);
  std::string token;
  while (std::getline(stream, token, ',')) {
    try {
      const int cpu = std::stoi(token);
      if (cpu >= 0 && cpu < CPU_SETSIZE) CPU_SET(cpu, &cpuset);
    } catch (...) {
      // Ignore malformed entries and keep the process on the default mask.
    }
  }
  if (CPU_COUNT(&cpuset) > 0) sched_setaffinity(0, sizeof(cpuset), &cpuset);
#else
  (void)cpu_list;
#endif
}

}  // namespace cable

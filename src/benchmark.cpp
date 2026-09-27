#include "cable_detector.hpp"
#include "json_writer.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace {

struct Options {
  int width{640};
  int height{480};
  int frames{120};
  int warmup{10};
  int interval{3};
  int threads{1};
  std::string input;
  std::string output_json;
  std::string save_image;
};

Options parse_options(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    auto value = [&](const char* name) -> std::string {
      if (arg == name && i + 1 < argc) return argv[++i];
      const std::string prefix = std::string(name) + "=";
      if (arg.rfind(prefix, 0) == 0) return arg.substr(prefix.size());
      return {};
    };
    if (const auto value_string = value("--width"); !value_string.empty()) options.width = std::stoi(value_string);
    else if (const auto value_string = value("--height"); !value_string.empty()) options.height = std::stoi(value_string);
    else if (const auto value_string = value("--frames"); !value_string.empty()) options.frames = std::stoi(value_string);
    else if (const auto value_string = value("--warmup"); !value_string.empty()) options.warmup = std::stoi(value_string);
    else if (const auto value_string = value("--interval"); !value_string.empty()) options.interval = std::stoi(value_string);
    else if (const auto value_string = value("--threads"); !value_string.empty()) options.threads = std::stoi(value_string);
    else if (const auto value_string = value("--input"); !value_string.empty()) options.input = value_string;
    else if (const auto value_string = value("--json"); !value_string.empty()) options.output_json = value_string;
    else if (const auto value_string = value("--save-image"); !value_string.empty()) options.save_image = value_string;
    else if (arg == "--help") {
      std::cout << "Usage: cable_benchmark [--input image] [--frames N] [--warmup N] "
                   "[--interval N] [--threads N] [--json report.json] [--save-image out.jpg]\n";
      std::exit(0);
    }
  }
  options.frames = std::max(1, options.frames);
  options.warmup = std::max(0, options.warmup);
  options.interval = std::max(1, options.interval);
  return options;
}

cv::Mat make_synthetic_frame(int width, int height, int frame_index) {
  cv::Mat image(height, width, CV_8UC3, cv::Scalar(220, 220, 220));
  cv::line(image, {20, height / 2 - 35}, {width - 20, height / 2 + 20},
           {28, 28, 28}, 45, cv::LINE_AA);
  cv::line(image, {20, height / 2 - 35}, {width - 20, height / 2 + 20},
           {55, 55, 55}, 2, cv::LINE_AA);
  const int cx = width / 2 + static_cast<int>(std::sin(frame_index * 0.07) * width * 0.12);
  const int cy = height / 2 - 8 + static_cast<int>(std::cos(frame_index * 0.05) * 4);
  cv::RotatedRect defect({static_cast<float>(cx), static_cast<float>(cy)}, {48.0F, 24.0F}, -8.0F);
  cv::Point2f corners[4];
  defect.points(corners);
  std::vector<cv::Point> polygon;
  for (const auto& point : corners) polygon.emplace_back(cvRound(point.x), cvRound(point.y));
  cv::fillConvexPoly(image, polygon, {210, 210, 210});
  cv::polylines(image, polygon, true, {255, 255, 255}, 3, cv::LINE_AA);
  cv::circle(image, {cx - 9, cy + 2}, 5, {245, 245, 245}, cv::FILLED, cv::LINE_AA);
  return image;
}

double mean(const std::vector<double>& values) {
  if (values.empty()) return 0.0;
  return std::accumulate(values.begin(), values.end(), 0.0) / values.size();
}

double percentile(std::vector<double> values, double p) {
  if (values.empty()) return 0.0;
  std::sort(values.begin(), values.end());
  const double index = (values.size() - 1) * p;
  const size_t lo = static_cast<size_t>(index);
  const size_t hi = std::min(values.size() - 1, lo + 1);
  return values[lo] + (values[hi] - values[lo]) * (index - lo);
}

}  // namespace

int main(int argc, char** argv) {
  const Options options = parse_options(argc, argv);
  cable::configure_opencv_threads(options.threads);
  cv::Mat input;
  if (!options.input.empty()) input = cv::imread(options.input, cv::IMREAD_COLOR);
  if (input.empty()) input = make_synthetic_frame(options.width, options.height, 0);

  cable::DetectorConfig config;
  config.inference_interval = options.interval;
  config.class_names = {"defect"};
  config.confidence_threshold = 0.50F;
  cable::CableDetector detector(config);

  for (int i = 0; i < options.warmup; ++i) detector.process(make_synthetic_frame(input.cols, input.rows, i));
  std::vector<double> timings;
  timings.reserve(options.frames);
  size_t detections = 0;
  int inference_frames = 0;
  cv::Mat last_annotated;
  for (int i = 0; i < options.frames; ++i) {
    cv::Mat frame = options.input.empty() ? make_synthetic_frame(input.cols, input.rows, i) : input;
    const auto start = std::chrono::steady_clock::now();
    const auto result = detector.process(frame);
    const auto end = std::chrono::steady_clock::now();
    timings.push_back(std::chrono::duration<double, std::milli>(end - start).count());
    detections += result.detections.size();
    inference_frames += result.inference_ran ? 1 : 0;
    last_annotated = result.annotated;
  }
  if (!options.save_image.empty() && !last_annotated.empty()) cv::imwrite(options.save_image, last_annotated);

  const double total_ms = std::accumulate(timings.begin(), timings.end(), 0.0);
  const double mean_ms = mean(timings);
  const double p95_ms = percentile(timings, 0.95);
  const double fps = mean_ms > 0.0 ? 1000.0 / mean_ms : 0.0;
  std::cout << std::fixed << std::setprecision(3)
            << "backend=" << detector.backend()->name() << "\n"
            << "frames=" << options.frames << " inference_frames=" << inference_frames
            << " reused_frames=" << detector.reused_frames() << " detections=" << detections << "\n"
            << "total_ms=" << total_ms << " mean_ms=" << mean_ms
            << " p95_ms=" << p95_ms << " fps=" << fps << "\n";

  if (!options.output_json.empty()) {
    std::ofstream report(options.output_json);
    report << "{\n"
           << "  \"backend\": \"" << detector.backend()->name() << "\",\n"
           << "  \"frames\": " << options.frames << ",\n"
           << "  \"warmup\": " << options.warmup << ",\n"
           << "  \"inference_interval\": " << options.interval << ",\n"
           << "  \"inference_frames\": " << inference_frames << ",\n"
           << "  \"reused_frames\": " << detector.reused_frames() << ",\n"
           << "  \"detections\": " << detections << ",\n"
           << "  \"total_ms\": " << total_ms << ",\n"
           << "  \"mean_ms\": " << mean_ms << ",\n"
           << "  \"p95_ms\": " << p95_ms << ",\n"
           << "  \"fps\": " << fps << "\n"
           << "}\n";
  }
  return 0;
}

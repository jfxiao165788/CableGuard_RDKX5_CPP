#pragma once

#include "cable_detector.hpp"

#include <iomanip>
#include <sstream>

namespace cable {

inline std::string json_escape(const std::string& value) {
  std::ostringstream out;
  for (const char c : value) {
    switch (c) {
      case '"': out << "\\\""; break;
      case '\\': out << "\\\\"; break;
      case '\n': out << "\\n"; break;
      case '\r': out << "\\r"; break;
      case '\t': out << "\\t"; break;
      default: out << c; break;
    }
  }
  return out.str();
}

inline std::string detections_json(const std::vector<Detection>& detections) {
  std::ostringstream out;
  out << "[";
  for (size_t i = 0; i < detections.size(); ++i) {
    const auto& d = detections[i];
    if (i != 0) out << ",";
    out << "{\"class\":\"" << json_escape(d.class_name)
        << "\",\"confidence\":" << std::fixed << std::setprecision(6)
        << d.confidence << ",\"bbox\":[" << d.bbox.x1 << ","
        << d.bbox.y1 << "," << d.bbox.x2 << "," << d.bbox.y2
        << "],\"center\":[" << d.center.x << "," << d.center.y
        << "],\"refined\":" << (d.refined ? "true" : "false");
    if (d.mask_ratio > 0.0F) out << ",\"mask_ratio\":" << d.mask_ratio;
    if (d.refine_area > 0.0F) out << ",\"refine_area\":" << d.refine_area;
    out << "}";
  }
  out << "]";
  return out.str();
}

}  // namespace cable

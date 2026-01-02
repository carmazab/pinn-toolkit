#pragma once

#include <yaml-cpp/yaml.h>

#include <string>

namespace io {
struct Node {
  explicit Node(const YAML::Node& node) : node_{node} {}

  template <typename T>
  T as() const {
    return node_.as<T>();
  }

  Node operator[](const std::string& key) const { return Node{node_[key]}; }

 private:
  YAML::Node node_;
};

Node load_inputfile(const std::string& filename) {
  return Node{YAML::LoadFile(filename)};
}
}  // namespace io
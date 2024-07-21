#include <filesystem>

constexpr bool kEnableDebugging = false;

#define DEBUG_SIZE(var) \
  if constexpr(kEnableDebugging) {std::cerr << #var << " size: " << var.sizes() << std::endl; }; 

#define SAVE_TENSOR(var) \
  if constexpr (kEnableDebugging) { \
    std::filesystem::create_directories("debug/" + std::string(__func__)); \
    torch::save(var, "debug/" + std::string(__func__) + "/" + std::string(#var) + ".pt"); \
  }

#define DEBUG_TENSOR(var) {DEBUG_SIZE(var); SAVE_TENSOR(var);};
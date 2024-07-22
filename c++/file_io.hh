#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include <torch/torch.h>
#include <Eigen/Dense>


std::vector<std::vector<double>> LoadData(const std::string& filename) {
  std::vector<std::vector<double>> data;
  std::ifstream file(filename);
  std::string line;
  
  if (!file.is_open()) {
    std::cerr << "Unable to open file" << std::endl;
    return data;
  }
  
  while (std::getline(file, line)) {
    std::vector<double> row;
    std::stringstream ss(line);
    double value;
    
    while (ss >> value) {
      row.push_back(value);
    }
    
    data.push_back(row);
  }
  
  file.close();
  return data;
}


template <typename InternalType>
void DumpTypedTensorToCSV(const torch::Tensor& tensor, const std::string& filename) {
  std::ofstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error("Could not open file for writing.");
  }

  auto sizes = tensor.sizes();
  const auto dim = sizes.size();

  if(dim == 1) {
    for(int i = 0; i < sizes[0]; ++i) {
      file << tensor[i].item<InternalType>() << std::endl;
    }
  } else if(dim == 2) {
    for (int i = 0; i < sizes[0]; ++i) {
      for (int j = 0; j < sizes[1]; ++j) {
        file << tensor[i][j].item<InternalType>();
        if (j < sizes[1] - 1) {
          file << ",";
        }
      }
      file << "\n";
    }
  } else {
    throw std::runtime_error("Expected a 1D or 2D tensor.");
  }

  file.close();
}

void DumpTensorToCSV(const torch::Tensor& tensor, const std::string& filename) {
  switch(tensor.scalar_type()) {
    case torch::kFloat32:
      DumpTypedTensorToCSV<float>(tensor, filename);
      break;
    case torch::kFloat64:
      DumpTypedTensorToCSV<double>(tensor, filename);
      break;
    case torch::kInt32:
      DumpTypedTensorToCSV<int32_t>(tensor, filename);
      break;
    case torch::kInt64:
      DumpTypedTensorToCSV<int64_t>(tensor, filename);
      break;
    default:
      throw std::runtime_error("Unsupported tensor scalar type");
  }
}


torch::Tensor ToTensor(const std::vector<std::vector<double>>& input, torch::TensorOptions options)
{
  const int rows = input.size(); 
  const int cols = input.at(0).size();
  torch::Tensor result = torch::zeros({rows, cols}, options);
  
  for(int i = 0; i < rows; ++i) {
    for(int j = 0; j < cols; ++j) {
      result.index({i, j}) = input.at(i).at(j);
    }
  }

  return result;
}


template <typename Derived>
torch::Tensor ToTensor(const Eigen::MatrixBase<Derived>& eigen_matrix, torch::TensorOptions options) {
  using Scalar = typename Derived::Scalar;
  using Matrix = Eigen::Matrix<Scalar, Derived::ColsAtCompileTime, Derived::RowsAtCompileTime>;

  torch::Dtype dtype;
  if (std::is_same<Scalar, float>::value) {
      dtype = torch::kFloat32;
  } else if (std::is_same<Scalar, double>::value) {
      dtype = torch::kFloat64;
  } else if (std::is_same<Scalar, int>::value) {
      dtype = torch::kInt32;
  } else if (std::is_same<Scalar, long>::value) {
      dtype = torch::kInt64;
  } else {
      throw std::invalid_argument("Unsupported data type");
  }
  
  Matrix eigen_matrix_transposed = eigen_matrix.transpose();
  torch::TensorOptions in_options;
  in_options = in_options.dtype(dtype);
  torch::Tensor tensor = torch::from_blob(
    const_cast<double*>(eigen_matrix_transposed.data()),
    {eigen_matrix_transposed.rows(), eigen_matrix_transposed.cols()},
    in_options.dtype(dtype)
  );
  return tensor.clone().to(options);
}

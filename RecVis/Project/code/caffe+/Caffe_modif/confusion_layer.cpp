#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/confusion_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void ConfusionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int K_ = bottom[0]->count() / bottom[0]->num();
    // Intialize the weight
    vector<int> weight_shape(2);
    weight_shape[0] = K_;
    weight_shape[1] = K_;
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    // fill the weights
    shared_ptr<Filler<Dtype> > matrix_filler(GetFiller<Dtype>(
        this->layer_param_.confusion_param().matrix_filler()));
    matrix_filler->Fill(this->blobs_[0].get());
    // parameter initialization
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void ConfusionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Figure out the dimensions
  const int new_K = bottom[0]->count() / bottom[0]->num();
  CHECK_EQ(K_, new_K)
      << "Input size incompatible with inner product parameters.";
  vector<int> top_shape = bottom[0]->shape();
  top_shape.resize(1);
  top_shape[1] = K_;
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void ConfusionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const Dtype* weight = this->blobs_[0]->cpu_data();
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, K_, 1 , K_, (Dtype)1.,
      bottom_data, weight, (Dtype)0., top_data);
}

template <typename Dtype>
void ConfusionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (this->param_propagate_down_[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    const Dtype* bottom_data = bottom[0]->cpu_data();
    // Gradient with respect to weight
    caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, 1, K_, K_, (Dtype)1.,
        top_diff, bottom_data, (Dtype)1., this->blobs_[0]->mutable_cpu_diff());
  }
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    // Gradient with respect to bottom data
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, K_, K_, 1, (Dtype)1.,
        top_diff, this->blobs_[0]->cpu_data(), (Dtype)0.,
        bottom[0]->mutable_cpu_diff());
  }

  // Normalize the weight matrix
  const Dtype* OldWeight = this->blobs_[0]->cpu_data();
  Dtype* NewWeight;
  caffe_abs(K_*K_,OldWeight,NewWeight);

  for (int i = 0; i < K_; ++i) {
    int scale=0;
     for (int j = 0; j < K_; ++j) {
      scale+=NewWeight[i * K_ + j];
     }
     for (int j = 0; j < K_; ++j) {
      NewWeight[i * K_ + j]/=scale;
     }
  }
}

#ifdef CPU_ONLY
STUB_GPU(ConfusionLayer);
#endif

INSTANTIATE_CLASS(ConfusionLayer);
REGISTER_LAYER_CLASS(Confusion);

}  // namespace caffe

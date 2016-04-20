/*
 *  Copyright (c) 2016, Kai Kang (myfavouritekk@gmail.com). All rights reserved.
 */


#ifndef VIDEO_ATTRIB_DATA_LAYER_HPP_
#define VIDEO_ATTRIB_DATA_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "pthread.h"
#include "boost/scoped_ptr.hpp"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

using std::string;

namespace caffe {

template <typename Dtype>
void* VideoAttribDataLayerPrefetch(void* layer_pointer);

template <typename Dtype>
class VideoAttribDataLayer : public Layer<Dtype> {
  // The function used to perform prefetching.
  friend void* VideoAttribDataLayerPrefetch<Dtype>(void* layer_pointer);

 public:
  explicit VideoAttribDataLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual ~VideoAttribDataLayer();
  virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

 protected:
  virtual Dtype Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual Dtype Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom) { return; }
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom) { return; }

  virtual void CreatePrefetchThread();
  virtual void JoinPrefetchThread();
  virtual unsigned int PrefetchRand();

  shared_ptr<Caffe::RNG> prefetch_rng_;
  vector<string> file_list_;
  vector<int> start_frm_list_;
  vector<vector<int> > attrib_label_list_;
  vector<int> shuffle_index_;
  int lines_id_;

  int datum_channels_;
  int datum_length_;
  int datum_height_;
  int datum_width_;
  int datum_size_;
  int num_attrib_; // number of attributes
  pthread_t thread_;
  shared_ptr<Blob<Dtype> > prefetch_data_;
  shared_ptr<Blob<Dtype> > prefetch_label_;
  Blob<Dtype> data_mean_;
  bool output_labels_;
  Caffe::Phase phase_;
};

}



#endif /* VIDEO_ATTRIB_DATA_LAYER_HPP_ */

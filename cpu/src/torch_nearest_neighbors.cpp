
#include <torch/extension.h>
#include "ball_query.h"
#include "compat.h"
#include "neighbors.h"
#include "neighbors.cpp"
#include <iostream>


std::pair<at::Tensor, at::Tensor> ball_query(at::Tensor query,
					     at::Tensor support,
					     float radius, int max_num, int mode){

	at::Tensor out;
	at::Tensor out_dists;
	std::vector<long> neighbors_indices;

	auto options = torch::TensorOptions().dtype(torch::kLong).device(torch::kCPU);
	auto options_dist = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
	int max_count = 0;

	std::vector<float> neighbors_dists;

	AT_DISPATCH_ALL_TYPES(query.scalar_type(), "radius_search", [&] {


	auto data_q = query.DATA_PTR<scalar_t>();
	auto data_s = support.DATA_PTR<scalar_t>();
	std::vector<scalar_t> queries_stl = std::vector<scalar_t>(data_q,
								   data_q + query.size(0)*query.size(1));
	std::vector<scalar_t> supports_stl = std::vector<scalar_t>(data_s,
								   data_s + support.size(0)*support.size(1));

	max_count = nanoflann_neighbors<scalar_t>(queries_stl,
						  supports_stl,
						  neighbors_indices,
						  neighbors_dists,
						  radius,
						  max_num,
						  mode);
	});

	auto neighbors_dists_ptr = neighbors_dists.data();
	long* neighbors_indices_ptr = neighbors_indices.data();
	if(mode == 0){
		out = torch::from_blob(neighbors_indices_ptr, {query.size(0), max_count}, options=options);
		out_dists = torch::from_blob(neighbors_dists_ptr,
	 				     {query.size(0), max_count},
	 				     options=options_dist);

	}
	else if(mode ==1){
		out = torch::from_blob(neighbors_indices_ptr, {(int)neighbors_indices.size()/2, 2}, options=options);
		out_dists = torch::from_blob(neighbors_dists_ptr,
					     {(int)neighbors_indices.size()/2, 1},
					     options=options_dist);
	}

	return std::make_pair(out.clone(), out_dists.clone());
}

void cumsum(const vector<long>& batch, vector<long>& res){

	res.resize(batch[batch.size()-1]-batch[0]+1, 0);
	long ind = batch[0];
	long incr = 1;
	for(int i=1; i < batch.size(); i++){

		if(batch[i] == ind)
			incr++;
		else{
			res[ind-batch[0]] = incr;
			incr =1;
			ind = batch[i];
		}
	}
	res[ind-batch[0]] = incr;
}

std::pair<at::Tensor, at::Tensor> batch_ball_query(at::Tensor query,
						   at::Tensor support,
						   at::Tensor query_batch,
						   at::Tensor support_batch,
						   float radius, int max_num, int mode) {
	at::Tensor out;
	at::Tensor out_dists;
	auto data_qb = query_batch.DATA_PTR<long>();
	auto data_sb = support_batch.DATA_PTR<long>();
	std::vector<long> query_batch_stl = std::vector<long>(data_qb, data_qb+query_batch.size(0));
	std::vector<long> cumsum_query_batch_stl;
	cumsum(query_batch_stl, cumsum_query_batch_stl);
	std::vector<long> support_batch_stl = std::vector<long>(data_sb, data_sb+support_batch.size(0));
	std::vector<long> cumsum_support_batch_stl;
	cumsum(support_batch_stl, cumsum_support_batch_stl);
	std::vector<long> neighbors_indices;

	auto options = torch::TensorOptions().dtype(torch::kLong).device(torch::kCPU);
	auto options_dist = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
	int max_count = 0;
	std::vector<float> neighbors_dists;
	AT_DISPATCH_ALL_TYPES(query.scalar_type(), "batch_radius_search", [&] {

	auto data_q = query.DATA_PTR<scalar_t>();
	auto data_s = support.DATA_PTR<scalar_t>();
	std::vector<scalar_t> queries_stl = std::vector<scalar_t>(data_q,
								  data_q + query.size(0)*query.size(1));
	std::vector<scalar_t> supports_stl = std::vector<scalar_t>(data_s,
								   data_s + support.size(0)*support.size(1));

	max_count = batch_nanoflann_neighbors<scalar_t>(queries_stl,
							supports_stl,
							cumsum_query_batch_stl,
							cumsum_support_batch_stl,
							neighbors_indices,
							neighbors_dists,
							radius,
							max_num,
							mode);
	});
	long* neighbors_indices_ptr = neighbors_indices.data();
	auto neighbors_dists_ptr = neighbors_dists.data();


	if(mode == 0){
		out = torch::from_blob(neighbors_indices_ptr, {query.size(0), max_count}, options=options);
		out_dists = torch::from_blob(neighbors_dists_ptr,
	 				     {query.size(0), max_count},
	 				     options=options_dist);
	}
	else if(mode == 1){
		out = torch::from_blob(neighbors_indices_ptr, {(int)neighbors_indices.size()/2, 2}, options=options);
		out_dists = torch::from_blob(neighbors_dists_ptr,
					     {(int)neighbors_indices.size()/2, 1},
					     options=options_dist);
	}
	return std::make_pair(out.clone(), out_dists.clone());
}

#include "group_points.h"
#include "ball_query.h"

using namespace pybind11::literals;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("group_points", &group_points);

    m.def("ball_query",
	      &ball_query,
	      "compute the radius search of a point cloud using nanoflann"
	      "- support : a pytorch tensor of size N1 x 3, points where the neighboors are accessed from"
	      "- query : a pytorch tensor of size N2 x 3, centre of the balls"
	      "- radius : float number, size of the ball for the radius search."
	      "- max_num : int number, indicate the maximum of neaghbors allowed(if -1 then all the possible neighbors will be computed). "
	      "- mode : int number that indicate which format for the neighborhood"
	      "mode=0 mean a matrix of neighbors(-1 for shadow neighbors)"
	      "mode=1 means a matrix of edges of size Num_edge x 2"
	      "return a tensor of size N2 x M where M is either max_num or the maximum number of neighbors found if mode = 0, if mode=1 return a tensor of size Num_edge x 2 and return a tensor containing the squared distance of the neighbors",
	      "support"_a, "querry"_a, "radius"_a, "max_num"_a=-1, "mode"_a=0);

    m.def("batch_ball_query",
	      &batch_ball_query,
		"compute the radius search of a point cloud for each batch using nanoflann"
		"- support : a pytorch tensor of size N1 x 3, points where the neighboors are accessed from"
		"- query : a pytorch tensor of size N2 x 3, centre of the balls"
		"- support_batch: a pytorch tensor(long) contains indices of the batch of the support size N1"
	      "NB: the batch must be sorted"
		"- query_batch : a pytorch tensor(long) contains indices of the batch of the query size N2"
	      "NB : the batch must be sorted"
		"-radius: float number, size of the ball for the radius search."
		"- max_num : int number, indicate the maximum of neaghbors allowed(if -1 then all the possible neighbors wrt the radius will be computed)."
		"- mode : int number that indicate which format for the neighborhood"
		"mode=0 mean a matrix of neighbors(N1 for shadow neighbors)"
		"mode=1 means a matrix of edges of size Num_edge x 2"
	      "return a tensor of size N2 x M where M is either max_num or the maximum number of neighbors found if mode = 0, if mode=1 return a tensor of size Num_edge x 2 and return a tensor containing the squared distance of the neighbors",
	      "support"_a, "querry"_a, "query_batch"_a, "support_batch"_a, "radius"_a, "max_num"_a=-1, "mode"_a=0);
    m.def("dense_ball_query", &dense_ball_query,
	  "compute the radius search of a batch of point cloud using nanoflann"
		  "- support : a pytorch tensor of size B x N1 x 3, points where the neighboors are accessed from"
		  "- query : a pytorch tensor of size B x N2 x 3, centre of the balls"
	      "- radius : float number, size of the ball for the radius search."
	      "- max_num : int number, indicate the maximum of neaghbors allowed(if -1 then all the possible neighbors will be computed). "
	      "- mode : int number that indicate which format for the neighborhood"
	      "mode=0 mean a matrix of neighbors(-1 for shadow neighbors)"
	      "mode=1 means a matrix of edges of size Num_edge x 2"
	      "return a tensor of size B x N2 x M where M is either max_num or the maximum number of neighbors found if mode = 0, if mode=1 return a tensor of size Num_edge x 2 and return a tensor containing the squared distance of the neighbors",
	      "support"_a, "querry"_a, "radius"_a, "max_num"_a=-1, "mode"_a=0);
}

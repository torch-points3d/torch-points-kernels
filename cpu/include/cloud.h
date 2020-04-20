//
//
//		0==========================0
//		|    Local feature test    |
//		0==========================0
//
//		version 1.0 :
//			>
//
//---------------------------------------------------
//
//		Cloud header
//
//----------------------------------------------------
//
//		Hugues THOMAS - 10/02/2017
//

#pragma once

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <unordered_map>
#include <vector>

#include <time.h>

template <typename scalar_t> struct PointCloud
{
    void set(const std::vector<scalar_t>& new_pts)
    {
        pts = new_pts.data();
        length = new_pts.size() / 3;
    }
    void set_batch(const std::vector<scalar_t>& new_pts, int begin, int end)
    {
        pts = new_pts.data();
        int start = begin * 3;
        pts += start;
        length = (end - begin);
    }

    // Must return the number of data points
    inline size_t kdtree_get_point_count() const
    {
        return get_point_count();
    }

    // Must return the number of data points
    inline size_t get_point_count() const
    {
        return length;
    }

    // Returns the dim'th component of the idx'th point in the class:
    // Since this is inlined and the "dim" argument is typically an immediate
    // value, the
    //  "if/else's" are actually solved at compile time.
    inline scalar_t kdtree_get_pt(const size_t idx, const size_t dim) const
    {
        if (dim == 0)
            return pts[idx * 3];
        else if (dim == 1)
            return pts[idx * 3 + 1];
        else
            return pts[idx * 3 + 2];
    }

    // Optional bounding-box computation: return false to default to a standard
    // bbox computation loop.
    //   Return true if the BBOX was already computed by the class and returned in
    //   "bb" so it can be avoided to redo it again. Look at bb.size() to find out
    //   the expected dimensionality (e.g. 2 or 3 for point clouds)
    template <class BBOX> bool kdtree_get_bbox(BBOX& /* bb */) const
    {
        return false;
    }

    const scalar_t* get_point_ptr(const int i) const
    {
        return pts + i * 3;
    }

    std::array<scalar_t, 3> operator[](const size_t index) const
    {
        return {pts[index * 3], pts[index * 3 + 1], pts[index * 3 + 2]};
    }

private:
    const scalar_t* pts;
    size_t length;
};

template <typename scalar_t>
inline std::ostream& operator<<(std::ostream& os, const PointCloud<scalar_t>& P)
{
    for (size_t i = 0; i < P.get_point_count(); i++)
    {
        auto p = P[i];
        os << "[" << p[0] << ", " << p[1] << ", " << p[2] << "];";
    }
    return os;
}
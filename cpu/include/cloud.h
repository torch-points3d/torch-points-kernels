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
    struct PointXYZ
    {
        scalar_t x, y, z;
    };

    std::vector<PointXYZ> pts;

    void set(const std::vector<scalar_t>& new_pts)
    {
        pts.clear();
        pts.resize(new_pts.size() / 3);
        for (unsigned int i = 0; i < new_pts.size(); i++)
        {
            if (i % 3 == 0)
            {
                PointXYZ point;
                point.x = new_pts[i];
                point.y = new_pts[i + 1];
                point.z = new_pts[i + 2];
                pts[i / 3] = point;
            }
        }
    }
    void set_batch(const std::vector<scalar_t>& new_pts, int begin, int end)
    {
        int size = end - begin;
        pts.clear();
        pts.resize(size);
        for (int i = 0; i < size; i++)
        {
            PointXYZ point;
            point.x = new_pts[3 * (begin + i)];
            point.y = new_pts[3 * (begin + i) + 1];
            point.z = new_pts[3 * (begin + i) + 2];
            pts[i] = point;
        }
    }

    // Must return the number of data points
    inline size_t kdtree_get_point_count() const
    {
        return pts.size();
    }

    // Returns the dim'th component of the idx'th point in the class:
    // Since this is inlined and the "dim" argument is typically an immediate
    // value, the
    //  "if/else's" are actually solved at compile time.
    inline scalar_t kdtree_get_pt(const size_t idx, const size_t dim) const
    {
        if (dim == 0)
            return pts[idx].x;
        else if (dim == 1)
            return pts[idx].y;
        else
            return pts[idx].z;
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
};

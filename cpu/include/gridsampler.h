#pragma once
#include <torch/extension.h>
#include <unordered_map>

class GridSampler
{
public:
    GridSampler(float gridSize) : m_gridSize(gridSize), m_voxelMap({}), m_numFittedPoints(0){};
    void fit(at::Tensor points);
    at::Tensor aggregate(at::Tensor data, c10::optional<std::string> mode = "mean") const;

    float gridSize() const
    {
        return m_gridSize;
    };

private:
    float m_gridSize;
    std::unordered_map<size_t, std::vector<size_t>> m_voxelMap;
    size_t m_numFittedPoints;
};

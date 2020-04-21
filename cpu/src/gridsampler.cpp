#include <stdexcept>

#include "compat.h"
#include "gridsampler.h"
#include "utils.h"

size_t getVoxelIdx(const long* indPtr, const long* discreteRangePtr)
{
    return static_cast<size_t>(indPtr[0] + indPtr[1] * discreteRangePtr[0] +
                               indPtr[2] * discreteRangePtr[0] * discreteRangePtr[1]);
}

void GridSampler::fit(const at::Tensor& points)
{
    CHECK_CONTIGUOUS(points);
    CHECK_CPU(points);
    TORCH_CHECK(points.size(1) == 3 && points.dim() == 2,
                "Input Tensor points must be of shape [N,3]")

    if (!m_voxelMap.empty())
        m_voxelMap.clear();
    m_numFittedPoints = static_cast<size_t>(points.size(0));
    m_voxelMap.reserve(m_numFittedPoints);
    m_voxels.reserve(m_numFittedPoints);

    // Centre to origin
    auto start = std::get<0>(points.min(0));
    auto end = std::get<0>(points.max(0));
    auto discreteRange = ((end - start) / m_gridSize).toType(torch::kLong) + 1;
    auto originCentred = points - start.unsqueeze(0);
    auto indices = (originCentred / m_gridSize).toType(torch::kLong);

    // Prepare for loop
    long* indicesPtr = indices.DATA_PTR<long>();
    long* discreteRangePtr = discreteRange.DATA_PTR<long>();
    const auto numPoints = static_cast<size_t>(points.size(0));
    for (size_t ptIdx = 0; ptIdx < numPoints; ptIdx++)
    {
        const size_t voxelIdx = getVoxelIdx(indicesPtr, discreteRangePtr);
        auto voxel = m_voxelMap.find(voxelIdx);
        if (voxel == m_voxelMap.end())
        {
            m_voxelMap.insert({voxelIdx, m_voxels.size()});
            m_voxels.push_back({ptIdx});
        }
        else
            m_voxels[voxel->second].push_back(ptIdx);

        // Move to next point
        indicesPtr += 3;
    }
}

at::Tensor GridSampler::aggregate(const at::Tensor& data, c10::optional<std::string> mode) const
{
    // Check input
    CHECK_CONTIGUOUS(data);
    CHECK_CPU(data);
    TORCH_CHECK(data.dim() == 2 || data.dim() == 1,
                "Input Tensor points must be of dimension 1 or 2")
    TORCH_CHECK(static_cast<size_t>(data.size(0)) == m_numFittedPoints,
                "You are trying to aggregate data that don't have the same number of nodes as the "
                "point cloud that was used to created the grid.")

    if (!(mode.value() == "mean" || mode.value() == "first" || mode.value() == "max_count"))
        throw std::invalid_argument("The mode argument can only be mean or first or max_count.");

    if (mode.value() == "max_count")
        CHECK_INT_TENSOR(data);

    if (mode.value() == "mean")
        CHECK_FLOAT_TENSOR(data);

    // Aggregate values
    size_t numFeats;
    at::Tensor outTensor;
    if (data.dim() == 1)
    {
        numFeats = 1;
        outTensor = torch::zeros({static_cast<long>(m_voxels.size())}, data.options());
    }
    else
    {
        numFeats = static_cast<size_t>(data.size(1));
        outTensor = torch::zeros({static_cast<long>(m_voxels.size()), static_cast<long>(numFeats)},
                                 data.options());
    }
    long rawResIdx, rawSourceIdx;
    AT_DISPATCH_ALL_TYPES(data.scalar_type(), "_aggregate", [&] {
        auto outPtr = outTensor.DATA_PTR<scalar_t>();
        auto dataPtr = data.DATA_PTR<scalar_t>();
        size_t resIdx = 0;
        for (auto& sourceIdx : m_voxels)
        {
            for (size_t i = 0; i < numFeats; i++)
            {
                rawResIdx = resIdx * numFeats + i;
                if (mode.value() == "max_count")
                {
                    std::map<scalar_t, size_t> counts;
                    for (auto& idx : sourceIdx)
                    {
                        rawSourceIdx = idx * numFeats + i;
                        scalar_t fieldValue = dataPtr[rawSourceIdx];

                        auto itemCountIt = counts.find(fieldValue);
                        if (itemCountIt == counts.end())
                            counts.insert({fieldValue, 1});
                        else
                            itemCountIt->second += 1;
                    }
                    auto x = std::max_element(counts.begin(), counts.end(),
                                              [](const std::pair<scalar_t, size_t>& p1,
                                                 const std::pair<scalar_t, size_t>& p2) {
                                                  return p1.second < p2.second;
                                              });
                    outPtr[rawResIdx] = x->first;
                }
                else
                {
                    for (auto idx : sourceIdx)
                    {
                        rawSourceIdx = idx * numFeats + i;
                        outPtr[rawResIdx] += dataPtr[rawSourceIdx];
                        if (mode.value() == "first")
                            break;
                    }
                    if (mode.value() == "mean")
                        outPtr[rawResIdx] /= static_cast<float>(sourceIdx.size());
                }
            }
            resIdx++;
        }
    });
    return outTensor;
}

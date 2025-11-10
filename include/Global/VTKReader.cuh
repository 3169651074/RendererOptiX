#ifndef RENDEREROPTIX_VTKREADER_CUH
#define RENDEREROPTIX_VTKREADER_CUH

#include <Global/RendererImpl.cuh>

namespace project {
    //VTK粒子
    typedef struct VTKParticle {
        //粒子ID
        size_t id;

        //速度
        float3 velocity;

        //包围盒范围
        float2 boundingBoxRanges[3];

        //质心
        float3 centroid;

        //组成此粒子的所有三角形顶点，保留triangle strip的格式
        std::vector<float3> vertices;

        //每个顶点对应的法向量，和顶点一一对应
        std::vector<float3> verticesNormals;
    } VTKParticle;

    class VTKReader {
    public:
        //读取.series文件，返回包含每个VTK文件路径和其时间的数组，数组长度为VTK文件数
        static std::tuple<std::vector<std::string>, std::vector<float>, size_t> readSeriesFile(const std::string & filePath);

        //读取单个VTK文件所有粒子
        static std::vector<VTKParticle> readVTKFile(const std::string & filePath);

        //将VTK粒子原始信息转换为粒子数组
        static std::vector<Particle> convertToRendererData(const std::vector<VTKParticle> & particles);
    };
}

#endif //RENDEREROPTIX_VTKREADER_CUH
